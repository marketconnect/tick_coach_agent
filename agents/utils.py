import os
import logging
import httpx

log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SERVICE_NAME = "tick-coach-agent"

def send_telegram_error_notification(client_id: str, model_name: str, error_message: str):
    """
    Sends a formatted error message to a Telegram chat.
    Failures are logged and suppressed.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram notification credentials not set. Skipping notification.")
        return

    message = (
        f"🚨 **Error in {SERVICE_NAME}** 🚨\n\n"
        f"**Client ID:** `{client_id}`\n"
        f"**Model Name:** `{model_name}`\n\n"
        f"**Error Details:**\n"
        f"```\n{error_message}\n```"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        # Use a separate, simple httpx client for notifications to avoid proxy issues if any.
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
    except Exception as e:
        log.error(f"Failed to send Telegram error notification: {e}", exc_info=True)