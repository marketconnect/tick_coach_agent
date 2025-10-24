import json, base64, logging, sys
from agents.message_handler import process_user_message

# --- BEGIN LangGraph compatibility shim (must be first) ---
import sys, types
try:
    # Canonical location since LangGraph 0.1.0+: langgraph.errors.GraphRecursionError
    from langgraph.errors import GraphRecursionError as _GraphRecursionError
    # Provide backward-compat alias for libs that still import from langgraph.pregel
    pregel_mod = types.ModuleType("langgraph.pregel")
    pregel_mod.GraphRecursionError = _GraphRecursionError
    sys.modules.setdefault("langgraph.pregel", pregel_mod)
except Exception:
    # If older LangGraph is installed, do nothing (it may already expose the old path)
    pass
# --- END LangGraph compatibility shim ---


# Configure logging to output to stdout, which will be captured by Yandex Cloud Function logs.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)


def main(event, context):
    """
    Универсальный обработчик для интеграции WebSocket API Gateway:
    - CONNECT/DISCONNECT: 200 OK
    - MESSAGE: вызывает обработчик сообщения агента и отправляет структурированный JSON-ответ.
    """
    logging.info(f"Function invoked. Event: {json.dumps(event, indent=2)}")
    rc = (event or {}).get("requestContext", {})
    ws = rc.get("websocket") or {}
    etype = ws.get("eventType") or rc.get("eventType")  # CONNECT | MESSAGE | DISCONNECT

    if etype in ["CONNECT", "DISCONNECT"]:
        logging.info(f"Handling {etype} event. Exiting.")
        return {"statusCode": 200}

    if etype != "MESSAGE":
        return {"statusCode": 400, "body": "Unsupported event type."}

    body_str = event.get("body", "")
    if event.get("isBase64Encoded"):
        try:
            body_str = base64.b64decode(body_str).decode("utf-8", "ignore")
        except Exception:
            logging.error("Failed to decode base64 body.", exc_info=True)
            body_str = ""

    if not body_str:
        logging.error("Empty message body received.")
        return {"statusCode": 400, "body": "Empty message body."}

    try:
        data = json.loads(body_str)
        client_id = data.get("clientId")
        user_prompt = data.get("message")

        if not client_id or not user_prompt:
            logging.error(f"Missing 'clientId' or 'message' in payload: {body_str}")
            return {"statusCode": 400, "body": "Payload must include 'clientId' and 'message'."}

        connection_id = ws.get("connectionId", "N/A")
        logging.info(f"Processing message for clientId '{client_id}' from connectionId '{connection_id}': {user_prompt}")

        agent_response_dict = process_user_message(client_id=client_id, user_prompt=user_prompt)
        response_body = json.dumps(agent_response_dict, ensure_ascii=False)

        logging.info(f"Sending response to connectionId '{connection_id}' for clientId '{client_id}': {response_body}")
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json; charset=utf-8"},
            "body": response_body,
        }
    except json.JSONDecodeError:
        logging.error(f"Failed to parse JSON from body: {body_str}")
        return {"statusCode": 400, "body": "Invalid JSON format."}
    except Exception as e:
        # client_id might not be defined if JSON parsing fails, so log cautiously.
        client_id_for_log = locals().get("client_id", "unknown")
        logging.error(f"Unhandled error during message processing for clientId '{client_id_for_log}'", exc_info=True)
        error_response = {"type": "error", "payload": "An internal error occurred."}
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json; charset=utf-8"},
            "body": json.dumps(error_response, ensure_ascii=False),
        }