import os
import logging
from typing import Generator
import httpx
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from openai import APIStatusError
import os
os.environ["OPENAI_USE_CHAT_COMPLETIONS_API"] = "true"
from langchain_openai import ChatOpenAI

proxy_url = os.getenv("PROXY_URL") or ""
timeout = httpx.Timeout(120.0, connect=10.0, read=120.0, write=120.0)
http_client = httpx.Client(timeout=timeout, proxy=proxy_url) if proxy_url else httpx.Client(timeout=timeout)

research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report."""

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
)
models = [model_gpt_5, model_glm_4_5]

def process_user_message(user_prompt: str) -> Generator[str, None, None]:
    """
    Processes a user prompt by attempting to stream a response from a series of models, with fallback.
    Yields content chunks as they are generated.
    """
    for i, model in enumerate(models):
        print(f"--- Попытка {i + 1}/{len(models)} с моделью: {model.model_name} --- {proxy_url}")
        try:
            agent = create_deep_agent(
                model=model,
                system_prompt=research_instructions,
            )
            
            result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
            msgs = result.get("messages") if isinstance(result, dict) else None
            if msgs:
                last = msgs[-1]
                text = getattr(last, "content", None)
                if text is None and isinstance(last, dict):
                    text = last.get("content", "")
                if text:
                    yield text

            print(f"--- Успешный ответ от модели: {model.model_name} ---")
            return

        except APIStatusError as e:
            if 400 <= e.status_code < 600:
                logging.warning(f"Модель {model.model_name} завершилась с ошибкой HTTP {e.status_code}. Пробуем следующую модель. Ошибка: {e}")
                continue
            else:
                logging.error(f"Модель {model.model_name} завершилась с неисправимой ошибкой API: {e}")
                continue
        except Exception as e:
            logging.error(f"Произошла непредвиденная ошибка с моделью {model.model_name}: {e}. Пробуем следующую модель.")
            continue
        
    yield "Не удалось получить ответ ни от одной из доступных моделей."