import os, re
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request
from openai import OpenAI

# Slack app
bolt_app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
    process_before_response=True  # lets us ack fast
)

# OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
    if event.get("channel_type") != "im":
        return
    ack()
    user_text = event.get("text", "")
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_text}],
        temperature=0.2
    )
    say(completion.choices[0].message.content)

# Flask web server that Railway will run
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