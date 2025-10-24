import os
import logging
import json
from typing import TypedDict, Annotated, List
from operator import add
import httpx
from deepagents import create_deep_agent
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from openai import APIStatusError, APIConnectionError
from langgraph.graph import StateGraph, END
from langgraph.errors import GraphRecursionError
import xmlschema



from .prompts import RESEARCH_INSTRUCTIONS, GOAL_CLARIFIER_SUBAGENT_DESCRIPTION, GOAL_CLARIFIER_SUBAGENT
from .utils import send_telegram_error_notification
from .ydb_checkpointer import YDBCheckpointer

log = logging.getLogger(__name__)

os.environ["OPENAI_USE_CHAT_COMPLETIONS_API"] = "true"

# Custom exception for interruptions
class ClarificationInterrupt(Exception):
    def __init__(self, questions: List[str]):
        self.questions = questions
        super().__init__("Agent needs clarification from the user.")

# State definition
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add]

# Tools
tools = []
tool_map = {}

proxy_url = os.getenv("PROXY_URL") or ""
timeout = httpx.Timeout(120.0, connect=10.0, read=120.0, write=120.0)
http_client_proxy = httpx.Client(timeout=timeout, proxy=proxy_url) if proxy_url else None
http_client_no_proxy = httpx.Client(timeout=timeout)
http_client = http_client_proxy or http_client_no_proxy

research_instructions = RESEARCH_INSTRUCTIONS

# Load XSD schema for Form validation
FORM_XSD_PATH = os.path.join(os.path.dirname(__file__), "..", "xmls", "Form.xml")
try:
    form_schema = xmlschema.XMLSchema(FORM_XSD_PATH)
except Exception as e:
    log.error(f"Failed to load XML Schema from {FORM_XSD_PATH}: {e}. XML validation disabled.")
    form_schema = None

# DO NOT CHANGE THE ORDER AND STRUCTURE OF THESE MODELS
model_glm_4_5 = ChatOpenAI(
  api_key=os.getenv("OPENROUTER_API_KEY"),
  base_url=os.getenv("OPENROUTER_BASE_URL"),
  model="z-ai/glm-4.5-air:free",
  default_headers={
    "HTTP-Referer": "https://www.your-site.com",
    "X-Title": "Your Site",
  },
  http_client=http_client,
)


model_gpt_5 = ChatOpenAI(
    model="gpt-5",
    base_url="https://agentrouter.org/v1",
    openai_api_key=os.getenv("AGENTROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "https://github.com/RooVetGit/Roo-Cline",
        "X-Title": "Roo Code",
        "User-Agent": "RooCode/1.0.0",
    },
    # DO NOT ADD http_client=http_client, it breaks the proxy
)
models = [model_gpt_5, model_glm_4_5]
# DO NOT CHANGE THE ORDER AND STRUCTURE OF THESE MODELS ABOVE

def rebuild_model_without_proxy(model: ChatOpenAI) -> ChatOpenAI:
    """
    Rebuild the provided model instance to use a no-proxy HTTP client.
    Used to mitigate httpx.ProxyError / APIConnectionError in serverless runtime.
    """
    if getattr(model, "model_name", "") == "gpt-5":
        return ChatOpenAI(
            model="gpt-5",
            base_url="https://agentrouter.org/v1",
            openai_api_key=os.getenv("AGENTROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": "https://github.com/RooVetGit/Roo-Cline",
                "X-Title": "Roo Code",
                "User-Agent": "RooCode/1.0.0",
            },
            http_client=http_client_no_proxy,
        )
    elif getattr(model, "model_name", "") == "z-ai/glm-4.5-air:free":
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
            model="z-ai/glm-4.5-air:free",
            default_headers={
                "HTTP-Referer": "https://www.your-site.com",
                "X-Title": "Your Site",
            },
            http_client=http_client_no_proxy,
        )
    else:
        return model

checkpointer = YDBCheckpointer()

def validate_form_xml(xml_str: str) -> bool:
    """Validate clarifier XML against xmls/Form.xml XSD."""
    if form_schema is None:
        # If schema failed to load, skip strict validation to avoid blocking flow
        return True
    try:
        return form_schema.is_valid(xml_str)
    except Exception as e:
        log.error(f"XML validation error: {e}")
        return False

def get_last_task_xml(messages: List[AnyMessage]) -> str | None:
    """Extract the most recent XML-looking payload from the conversation (ToolMessage or AIMessage)."""
    for msg in reversed(messages):
        try:
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content.lstrip().startswith("<"):
                return content
        except Exception:
            continue
    return None

def _sanitize_messages_for_model(messages: List[AnyMessage]) -> List[AnyMessage]:
    """Remove ToolMessage entries to avoid invalid OpenAI Chat API history."""
    safe: List[AnyMessage] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            # Drop dangling tool messages from persisted history
            continue
        safe.append(m)
    return safe

# Graph nodes
def agent_node(state: AgentState, agent):
    # Sanitize any dangling tool messages from persisted history before invoking the model
    safe_state = dict(state)
    safe_state["messages"] = _sanitize_messages_for_model(state.get("messages", []))
    result = agent.invoke(safe_state)

    # If the inner agent produced an XML payload via a tool, surface it as an AI message (not a ToolMessage)
    msgs = result["messages"]
    xml_content = None
    for m in reversed(msgs):
        try:
            c = getattr(m, "content", None)
            if isinstance(c, str) and c.lstrip().startswith("<"):
                xml_content = c
                break
        except Exception:
            continue
    if xml_content is not None:
        return {"messages": [AIMessage(content=xml_content)]}

    # Fallback to the last message (prefer AIMessage if available)
    last = msgs[-1]
    if isinstance(last, ToolMessage):
        # Never persist a tool role in outer state; convert to assistant text
        return {"messages": [AIMessage(content=str(last.content))]}
    return {"messages": [last]}

def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END

    tool_calls = last_message.tool_calls

    # Execute only known local tools; ignore others (handled internally by deep agent)
    tool_outputs = []
    for tool_call in tool_calls:
        name = tool_call["name"]
        if name not in tool_map:
            continue
        tool_output = tool_map[name].invoke(tool_call["args"])
        tool_outputs.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))

    if not tool_outputs:
        return END
    return {"messages": tool_outputs}

# Conditional edge
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# Build graph
def build_graph(model):
    # Configure the goal clarifier as a SubAgent
    subagents = [
        {
            "name": "ask_clarifying_questions",
            "description": GOAL_CLARIFIER_SUBAGENT_DESCRIPTION,
            "system_prompt": GOAL_CLARIFIER_SUBAGENT,
        }
    ]

    agent = create_deep_agent(
        model=model,
        system_prompt=research_instructions,
        tools=tools,
        subagents=subagents,
    )

    graph = StateGraph(AgentState)
    graph.add_node("agent", lambda state: agent_node(state, agent))
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)

def process_user_message(client_id: str, user_prompt: str) -> dict:
    log.info(f"Starting message processing for client_id: {client_id}")
    thread_config = {"configurable": {"thread_id": client_id}}
    
    # The input to the graph is a dictionary with a list of messages.
    # LangGraph will automatically load the previous state, append this new message,
    # and then save the full state upon completion.
    graph_input = {"messages": [HumanMessage(content=user_prompt)]}

    for i, model in enumerate(models):
        log.info(f"--- Attempt {i + 1}/{len(models)} with model: {model.model_name} ---")
        try:
            graph = build_graph(model)
            final_state = graph.invoke(graph_input, thread_config)
            messages = final_state["messages"]
            xml_payload = get_last_task_xml(messages)
            if xml_payload:
                is_valid = validate_form_xml(xml_payload)
                if is_valid:
                    return {
                        "type": "clarifying_question",
                        "payload": xml_payload
                    }
                # Validation failed: request agent/subagent to regenerate valid XML (do not send invalid to user)
                fix_instruction = "XML из подагента ask_clarifying_questions не соответствует схеме Form.xml. Пересоздай валидный XML строго по схеме и выведи только XML."
                retry_state = graph.invoke({"messages": [HumanMessage(content=fix_instruction)]}, thread_config)
                retry_messages = retry_state["messages"]
                retry_xml = get_last_task_xml(retry_messages)
                if retry_xml and validate_form_xml(retry_xml):
                    return {
                        "type": "clarifying_question",
                        "payload": retry_xml
                    }
                return {
                    "type": "error",
                    "payload": "XML из подагента ask_clarifying_questions не валиден по схеме Form.xml. Попробуйте снова."
                }
            last_message = messages[-1]
            return {
                "type": "final_answer",
                "payload": last_message.content
            }
        except ClarificationInterrupt as e:
            log.info(f"--- Interruption for clarification from model: {model.model_name} ---")
            # The state is already saved by the checkpointer up to the point of interruption.
            return {
                "type": "clarifying_question",
                "payload": e.questions
            }
        except GraphRecursionError as e:
             logging.error(f"Модель {model.model_name} вошла в цикл. Пробуем следующую модель.")
             send_telegram_error_notification(client_id, model.model_name, f"GraphRecursionError: {e}")
             continue
        except APIStatusError as e:
            if 400 <= e.status_code < 600:
                logging.error(f"Модель {model.model_name} завершилась с ошибкой HTTP {e.status_code}. Пробуем следующую модель.", exc_info=True)
                send_telegram_error_notification(client_id, model.model_name, f"APIStatusError: {e}")
                continue
        except (httpx.ProxyError, APIConnectionError) as e:
            logging.error(f"Сетевая ошибка/прокси для модели {model.model_name}. Пытаемся без прокси.", exc_info=True)
            send_telegram_error_notification(client_id, model.model_name, f"{type(e).__name__}: {e}")
            try:
                model_no_proxy = rebuild_model_without_proxy(model)
                graph = build_graph(model_no_proxy)
                final_state = graph.invoke(graph_input, thread_config)
                messages = final_state["messages"]
                xml_payload = get_last_task_xml(messages)
                if xml_payload:
                    is_valid = validate_form_xml(xml_payload)
                    if is_valid:
                        return {
                            "type": "clarifying_question",
                            "payload": xml_payload
                        }
                    fix_instruction = "XML из подагента ask_clarifying_questions не соответствует схеме Form.xml. Пересоздай валидный XML строго по схеме и выведи только XML."
                    retry_state = graph.invoke({"messages": [HumanMessage(content=fix_instruction)]}, thread_config)
                    retry_messages = retry_state["messages"]
                    retry_xml = get_last_task_xml(retry_messages)
                    if retry_xml and validate_form_xml(retry_xml):
                        return {
                            "type": "clarifying_question",
                            "payload": retry_xml
                        }
                    return {
                        "type": "error",
                        "payload": "XML из подагента ask_clarifying_questions не валиден по схеме Form.xml. Попробуйте снова."
                    }
                last_message = messages[-1]
                return {
                    "type": "final_answer",
                    "payload": last_message.content
                }
            except Exception as e2:
                logging.error(f"Повтор без прокси для модели {model.model_name} завершился ошибкой.", exc_info=True)
                send_telegram_error_notification(client_id, model.model_name, f"Retry failed: {e2}")
                continue
        except Exception as e:
            logging.error(f"Произошла непредвиденная ошибка с моделью {model.model_name}. Пробуем следующую модель.", exc_info=True)
            send_telegram_error_notification(client_id, model.model_name, f"Unexpected error: {e}")
            continue
        
    log.error(f"All models failed for client_id: {client_id}")
    send_telegram_error_notification(client_id, "N/A", "All models failed.")
    return {
        "type": "error",
        "payload": "Не удалось получить ответ ни от одной из доступных моделей."
    }