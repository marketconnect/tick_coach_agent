import json, base64
from agents.message_handler import process_user_message


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

       
    if not body:
      return {"statusCode": 400, "body": "Empty message body."}
    
    # process_user_message returns a generator. We need to consume it
    # to get the full string response for the cloud function's return body.
    stream = process_user_message(user_prompt=body)
    agent_response = "".join(list(stream))

    return {
        "statusCode": 200,
        # "headers": {"Content-Type": content_type},
        # "body": message
        "headers": {"Content-Type": "text/plain; charset=utf-8"},
        "body": agent_response,
    }