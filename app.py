import os, re
import logging
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_sdk.errors import SlackApiError
from flask import Flask, request
from openai import OpenAI

# Required environment variables (fail fast with a clear error if missing)
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

_missing_env = [
    name for name, value in {
        "SLACK_BOT_TOKEN": SLACK_BOT_TOKEN,
        "SLACK_SIGNING_SECRET": SLACK_SIGNING_SECRET,
        "OPENAI_API_KEY": OPENAI_API_KEY,
    }.items() if not value
]
if _missing_env:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(_missing_env)}"
    )

# Slack app
try:
    # Disable token verification at startup to avoid import-time crashes in hosting envs.
    bolt_app = App(
        token=SLACK_BOT_TOKEN,
        signing_secret=SLACK_SIGNING_SECRET,
        process_before_response=True,  # lets us ack fast
        token_verification_enabled=False,
    )
except SlackApiError as e:
    raise RuntimeError(f"Slack API error during app init: {e}")

# OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def strip_mention(text: str) -> str:
    return re.sub(r"<@[^>]+>\s*", "", text or "").strip()

@bolt_app.event("app_mention")
def on_mention(body, say, ack):
    ack()  # respond to Slack within 3 seconds
    user_text = strip_mention(body["event"]["text"])
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_text}],
        temperature=0.2
    )
    say(completion.choices[0].message.content)

@bolt_app.event("message")
def on_dm(body, event, say, ack):
    # Ack immediately for all message events to prevent Slack retries
    ack()
    if event.get("channel_type") != "im":
        return
    user_text = event.get("text", "")
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_text}],
        temperature=0.2
    )
    say(completion.choices[0].message.content)

# Flask web server that Railway will run
logging.basicConfig(level=logging.INFO)
flask_app = Flask(__name__)
handler = SlackRequestHandler(bolt_app)

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

@flask_app.route("/healthz", methods=["GET"])
def health():
    return "ok", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    flask_app.run(host="0.0.0.0", port=port)