import os, re
import logging
import time
from collections import defaultdict
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_sdk.errors import SlackApiError
from flask import Flask, request
from openai import OpenAI
from anthropic import Anthropic

# Required environment variables (fail fast with a clear error if missing)
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# At least one AI service API key is required
_missing_env = [
    name for name, value in {
        "SLACK_BOT_TOKEN": SLACK_BOT_TOKEN,
        "SLACK_SIGNING_SECRET": SLACK_SIGNING_SECRET,
    }.items() if not value
]

if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
    _missing_env.append("OPENAI_API_KEY or ANTHROPIC_API_KEY")
    
_ai_services = []
if OPENAI_API_KEY:
    _ai_services.append("OpenAI")
if ANTHROPIC_API_KEY:
    _ai_services.append("Claude")
if _missing_env:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(_missing_env)}"
    )

print(f"AI services available: {', '.join(_ai_services)}")

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

# Initialize AI clients
openai_client = None
claude_client = None

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
if ANTHROPIC_API_KEY:
    claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Task detection tracking
task_suggestions = defaultdict(list)  # channel_id -> [timestamps]
SUGGESTION_COOLDOWN = 300  # 5 minutes between suggestions per channel

def strip_mention(text: str) -> str:
    return re.sub(r"<@[^>]+>\s*", "", text or "").strip()

def get_ai_response(user_text: str, prefer_claude: bool = True) -> str:
    """Get AI response, preferring Claude if available, falling back to OpenAI"""
    try:
        # Try Claude first if preferred and available
        if prefer_claude and claude_client:
            response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.2,
                messages=[{"role": "user", "content": user_text}]
            )
            return response.content[0].text
        
        # Try OpenAI if Claude not preferred or not available
        elif openai_client:
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": user_text}],
                temperature=0.2
            )
            return completion.choices[0].message.content
        
        # Try Claude as fallback if OpenAI failed and Claude is available
        elif claude_client:
            response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.2,
                messages=[{"role": "user", "content": user_text}]
            )
            return response.content[0].text
        
        else:
            return "Sorry, no AI service is currently available."
            
    except Exception as e:
        logging.error(f"AI response error: {e}")
        return "Sorry, I had trouble responding just now."

def detect_potential_task(message_text: str) -> bool:
    """Use AI to detect if a message suggests a potential task or action item"""
    if not message_text or len(message_text.strip()) < 10:
        return False
    
    # Quick pattern matching first
    task_patterns = [
        r'\b(we need to|someone should|todo|action item|can we|should we|let\'s)\b',
        r'\b(implement|fix|add|create|build|update|remove|delete)\b',
        r'\b(missing|broken|not working|issue|problem|bug)\b',
        r'\?(.*)(implement|fix|add|create|build|update)',
    ]
    
    text_lower = message_text.lower()
    if not any(re.search(pattern, text_lower) for pattern in task_patterns):
        return False
    
    # Use AI for more nuanced detection
    try:
        prompt = f"""Analyze this message and determine if it suggests a potential task, action item, or work that needs to be done.

Message: "{message_text}"

Return only "YES" if this message:
- Suggests something needs to be implemented, fixed, or created
- Mentions a problem that needs solving
- Contains action items or todos
- Asks about missing features or functionality

Return "NO" if it's just casual conversation, questions for information, or already completed work.

Answer:"""

        if claude_client:
            response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip().upper() == "YES"
        elif openai_client:
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            return completion.choices[0].message.content.strip().upper() == "YES"
    except Exception as e:
        logging.error(f"Task detection error: {e}")
    
    return False

def can_suggest_task(channel_id: str) -> bool:
    """Check if we can suggest a task (rate limiting)"""
    now = time.time()
    channel_suggestions = task_suggestions[channel_id]
    
    # Clean old suggestions (older than cooldown period)
    task_suggestions[channel_id] = [ts for ts in channel_suggestions if now - ts < SUGGESTION_COOLDOWN]
    
    # Allow suggestion if no recent ones
    return len(task_suggestions[channel_id]) == 0

def record_task_suggestion(channel_id: str):
    """Record that we made a task suggestion"""
    task_suggestions[channel_id].append(time.time())

@bolt_app.event("app_mention")
def on_mention(body, say, ack):
    ack()  # respond to Slack within 3 seconds
    logging.info("app_mention event received: %s", {
        "channel": body.get("event", {}).get("channel"),
        "user": body.get("event", {}).get("user"),
    })
    user_text = strip_mention(body["event"]["text"])
    
    # Check if user wants to specify which AI to use
    prefer_claude = True  # Default to Claude
    if user_text.lower().startswith("openai:") or user_text.lower().startswith("gpt:"):
        prefer_claude = False
        user_text = user_text.split(":", 1)[1].strip()
    elif user_text.lower().startswith("claude:"):
        prefer_claude = True
        user_text = user_text.split(":", 1)[1].strip()
    
    response = get_ai_response(user_text, prefer_claude=prefer_claude)
    say(response, thread_ts=body["event"]["ts"])

@bolt_app.event("message")
def on_message(body, event, say, ack):
    # Ack immediately for all message events to prevent Slack retries
    ack()
    logging.info("message event received: %s", {
        "channel_type": event.get("channel_type"),
        "channel": event.get("channel"),
        "user": event.get("user"),
    })
    
    # Handle DMs
    if event.get("channel_type") == "im":
        user_text = event.get("text", "")
        
        # Check if user wants to specify which AI to use
        prefer_claude = True  # Default to Claude
        if user_text.lower().startswith("openai:") or user_text.lower().startswith("gpt:"):
            prefer_claude = False
            user_text = user_text.split(":", 1)[1].strip()
        elif user_text.lower().startswith("claude:"):
            prefer_claude = True
            user_text = user_text.split(":", 1)[1].strip()
        
        response = get_ai_response(user_text, prefer_claude=prefer_claude)
        say(response)
        return
    
    # Handle channel messages for task detection
    if event.get("channel_type") in ["channel", "group"]:
        message_text = event.get("text", "")
        channel_id = event.get("channel")
        user_id = event.get("user")
        bot_user_id = event.get("bot_id")
        
        # Skip bot messages and messages without text
        if bot_user_id or not message_text or not channel_id:
            return
            
        # Skip if we can't suggest tasks yet (rate limiting)
        if not can_suggest_task(channel_id):
            return
            
        # Check if message suggests a potential task
        if detect_potential_task(message_text):
            record_task_suggestion(channel_id)
            
            suggestion_text = ("👋 It sounds like there might be a task here. "
                             "Should I help track this as an action item? "
                             "Just mention me to get started!")
            
            # Reply in thread to the original message
            try:
                say(suggestion_text, thread_ts=event.get("ts"))
            except Exception as e:
                logging.error(f"Failed to suggest task: {e}")

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