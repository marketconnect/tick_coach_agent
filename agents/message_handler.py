import os
import logging
import json
from typing import TypedDict, Annotated, List
from operator import add
import httpx
from deepagents import create_deep_agent
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from openai import APIStatusError, APIConnectionError
from langgraph.graph import StateGraph, END
from langgraph.errors import GraphRecursionError



from .tools import ask_clarifying_questions
from .prompts import RESEARCH_INSTRUCTIONS
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
tools = [ask_clarifying_questions]
tool_map = {tool.name: tool for tool in tools}

proxy_url = os.getenv("PROXY_URL") or ""
timeout = httpx.Timeout(120.0, connect=10.0, read=120.0, write=120.0)
http_client_proxy = httpx.Client(timeout=timeout, proxy=proxy_url) if proxy_url else None
http_client_no_proxy = httpx.Client(timeout=timeout)
http_client = http_client_proxy or http_client_no_proxy

research_instructions = RESEARCH_INSTRUCTIONS

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
    http_client=http_client,
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

# Graph nodes
def agent_node(state: AgentState, agent):
    result = agent.invoke(state)
    # We only want to add the new message to the state
    return {"messages": [result["messages"][-1]]}

def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END

    tool_calls = last_message.tool_calls
    
    for tool_call in tool_calls:
        if tool_call["name"] == "ask_clarifying_questions":
            raise ClarificationInterrupt(tool_call["args"]["questions"])

    tool_outputs = []
    for tool_call in tool_calls:
        tool_output = tool_map[tool_call["name"]].invoke(tool_call["args"])
        tool_outputs.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
    
    return {"messages": tool_outputs}

# Conditional edge
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# Build graph
def build_graph(model):
    agent = create_deep_agent(
        model=model,
        system_prompt=research_instructions,
        tools=tools,
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
            last_message = final_state["messages"][-1]
            
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
        except GraphRecursionError:
             logging.warning(f"Модель {model.model_name} вошла в цикл. Пробуем следующую модель.")
             continue
        except APIStatusError as e:
            if 400 <= e.status_code < 600:
                logging.warning(f"Модель {model.model_name} завершилась с ошибкой HTTP {e.status_code}. Пробуем следующую модель. Ошибка: {e}")
                continue
        except (httpx.ProxyError, APIConnectionError) as e:
            logging.warning(f"Сетевая ошибка/прокси для модели {model.model_name}: {e}. Пытаемся без прокси.")
            try:
                model_no_proxy = rebuild_model_without_proxy(model)
                graph = build_graph(model_no_proxy)
                final_state = graph.invoke(graph_input, thread_config)
                last_message = final_state["messages"][-1]
                return {
                    "type": "final_answer",
                    "payload": last_message.content
                }
            except Exception as e2:
                logging.warning(f"Повтор без прокси для модели {model.model_name} завершился ошибкой: {e2}. Пробуем следующую модель.")
                continue
        except Exception as e:
            logging.exception(f"Произошла непредвиденная ошибка с моделью {model.model_name}. Пробуем следующую модель.")
            continue
        
    log.warning(f"All models failed for client_id: {client_id}")
    return {
        "type": "error",
        "payload": "Не удалось получить ответ ни от одной из доступных моделей."
    }