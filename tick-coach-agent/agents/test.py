# %%
import os
from typing import Literal
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from openai import APIStatusError
import logging

# System prompt to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report."""

# %%
model_glm_4_5 = ChatOpenAI(
  api_key=os.getenv("OPENROUTER_API_KEY"),
  base_url=os.getenv("OPENROUTER_BASE_URL"),
  model="z-ai/glm-4.5-air:free",
  default_headers={
    "HTTP-Referer": "https://www.your-site.com",
    "X-Title": "Your Site",
  }
)


model_gpt_5 = ChatOpenAI(
    api_key=os.getenv("AGENTROUTER_API_KEY"),
    base_url="https://agentrouter.org/v1",
    model="gpt-5",
    default_headers={
        "HTTP-Referer": "https://github.com/RooVetGit/Roo-Cline",
        "X-Title": "Roo Code",
        "User-Agent": "RooCode/1.0.0",
    },
)

models = [model_glm_4_5, model_gpt_5]

# %%
def invoke_with_fallback(models_to_try, user_prompt, system_prompt):
    """
    Calls the deep agent with a list of models, switching to the next model in case of HTTP errors.
    """
    for i, model in enumerate(models_to_try):
        print(f"--- Попытка {i + 1}/{len(models_to_try)} с моделью: {model.model_name} ---")
        try:
            agent = create_deep_agent(
                model=model,
                system_prompt=system_prompt,
            )
            result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
            print(f"--- Успешный ответ от модели: {model.model_name} ---")
            return result
        except APIStatusError as e:
            if 400 <= e.status_code < 600:
                logging.warning(f"Модель {model.model_name} завершилась с ошибкой HTTP {e.status_code}. Пробуем следующую модель. Ошибка: {e}")
                continue
            else:
                logging.error(f"Модель {model.model_name} завершилась с неисправимой ошибкой API: {e}")
                raise e
        except Exception as e:
            logging.error(f"Произошла непредвиденная ошибка с моделью {model.model_name}: {e}. Пробуем следующую модель.")
            continue
        
    return {"messages": [{"role": "assistant", "content": "Не удалось получить ответ ни от одной из доступных моделей."}]}

# %%
user_task = "Создай мне тренировку."
result = invoke_with_fallback(
    models_to_try=models,
    user_prompt=user_task,
    system_prompt=research_instructions
)

# %%
# from IPython.display import Image, display

# # Create the deep agent
# agent = create_deep_agent(
#     model=models.get("gpt-5"),
#     # tools=[internet_search],
#     system_prompt=research_instructions,
# )

# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# %%
# Invoke the agent
# result = agent.invoke({"messages": [{"role": "user", "content": "Создай мне тренировку."}]})

# %%
from utils.utils import format_messages

format_messages(result["messages"])

# %%
result

# %%


