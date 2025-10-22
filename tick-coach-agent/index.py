import json, base64
from agents.test import process_user_message

# message = """{
#   "type": "workout",
#   "workout": {
#     "title": "Утренняя зарядка",
#     "set_count": 2,
#     "intervals": [
#       {
#         "kind": "prepare",
#         "title": "Подготовка",
#         "durationSec": 10
#       },
#       {
#         "kind": "work",
#         "title": "Прыжки на месте",
#         "description": "Держите легкий темп",
#         "durationSec": 45,
#         "isRepsBased": false
#       },
#       {
#         "kind": "rest",
#         "title": "Отдых",
#         "durationSec": 15
#       },
#       {
#         "kind": "work",
#         "title": "Приседания",
#         "reps": 20,
#         "isRepsBased": true
#       },
#       {
#         "kind": "between_sets",
#         "title": "Отдых между сетами",
#         "durationSec": 60
#       }
#     ]
#   }
# }"""

def main(event, context):
    """
    Универсальный обработчик для интеграции WebSocket API Gateway:
    - CONNECT/DISCONNECT: 200 OK
    - MESSAGE: вернуть то, что пришло (эхо).
      Тело ответа функции на MESSAGE отправляется клиенту отдельным WS-сообщением.
    """
    rc = (event or {}).get("requestContext", {})
    ws = rc.get("websocket") or {}
    etype = ws.get("eventType") or rc.get("eventType")  # CONNECT | MESSAGE | DISCONNECT

    # Тело запроса может быть base64
    body = event.get("body", "")
    if event.get("isBase64Encoded"):
        try:
            body = base64.b64decode(body).decode("utf-8", "ignore")
        except Exception:
            body = ""

    if etype == "CONNECT":
        return {"statusCode": 200}

    if etype == "DISCONNECT":
        return {"statusCode": 200}

    # MESSAGE: эхо — вернём как строку (или JSON, если это валидный JSON)
    # try:
    #     parsed = json.loads(body) if body else {}
    #     resp_body = json.dumps({"echo": parsed}, ensure_ascii=False)
    #     content_type = "application/json"
    # except Exception:
    #     resp_body = body
    #     content_type = "text/plain; charset=utf-8"
    
    if not body:
      return {"statusCode": 400, "body": "Empty message body."}
    
    agent_response = process_user_message(user_prompt=body)

    return {
        "statusCode": 200,
        # "headers": {"Content-Type": content_type},
        # "body": message
        "headers": {"Content-Type": "text/plain; charset=utf-8"},
        "body": agent_response,
    }