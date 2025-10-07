import json, base64

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
        return {"statusCode": 200}  # ничего не отсылаем в клиент

    if etype == "DISCONNECT":
        return {"statusCode": 200}

    # MESSAGE: эхо — вернём как строку (или JSON, если это валидный JSON)
    try:
        parsed = json.loads(body) if body else {}
        resp_body = json.dumps({"echo": parsed}, ensure_ascii=False)
        content_type = "application/json"
    except Exception:
        resp_body = body
        content_type = "text/plain; charset=utf-8"

    return {
        "statusCode": 200,
        "headers": {"Content-Type": content_type},
        "body": resp_body
    }