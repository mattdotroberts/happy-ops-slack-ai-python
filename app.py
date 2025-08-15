import os, re
import logging
import time
import requests
import asyncio
from collections import defaultdict
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_sdk.errors import SlackApiError
from flask import Flask, request
from openai import OpenAI
from anthropic import Anthropic

# Try to import MCP, but gracefully handle if not available
try:
    import mcp
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamablehttp_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP not available, will use direct API only")

# Required environment variables (fail fast with a clear error if missing)
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
NOTION_ACCESS_TOKEN = os.environ.get("NOTION_ACCESS_TOKEN")
SMITHERY_API_KEY = os.environ.get("SMITHERY_API_KEY")
SMITHERY_PROFILE = os.environ.get("SMITHERY_PROFILE")

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

# Notion configuration
NOTION_API_URL = "https://api.notion.com/v1"
NOTION_MCP_URL = "https://mcp.notion.com/sse"
SMITHERY_MCP_URL = "https://server.smithery.ai/@smithery/notion/mcp"
NOTION_VERSION = "2022-06-28"

# Global state
notion_session = None
notion_mcp_available = False

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

async def init_notion_mcp():
    """Initialize Notion MCP connection - tries Smithery first, then official MCP"""
    global notion_session, notion_mcp_available
    
    if not MCP_AVAILABLE:
        logging.info("MCP library not available, using direct API only")
        return False
    
    # Try Smithery MCP first if configured
    if SMITHERY_API_KEY and SMITHERY_PROFILE:
        try:
            logging.info("Attempting connection to Smithery Notion MCP server using streamable HTTP")
            
            # Construct URL exactly as shown in Smithery SDK example
            smithery_url = f"{SMITHERY_MCP_URL}?api_key={SMITHERY_API_KEY}&profile={SMITHERY_PROFILE}"
            
            # Use the exact pattern from Smithery's SDK example
            read_stream, write_stream, _ = await streamablehttp_client(smithery_url).__aenter__()
            session = mcp.ClientSession(read_stream, write_stream)
            await session.__aenter__()
            
            # Initialize the connection
            await session.initialize()
            
            # Test available tools
            tools_result = await session.list_tools()
            logging.info(f"✅ Smithery MCP connected! Available tools: {[tool.name for tool in tools_result.tools]}")
            
            notion_session = session
            notion_mcp_available = True
            return True
            
        except Exception as e:
            logging.warning(f"Smithery MCP connection failed: {e}")
            logging.info("Falling back to other connection methods")
    
    # Fallback to official Notion MCP if we have a token
    if NOTION_ACCESS_TOKEN:
        try:
            logging.info("Attempting connection to official Notion MCP server")
            headers = {
                "Authorization": f"Bearer {NOTION_ACCESS_TOKEN}",
                "Content-Type": "application/json"
            }
            
            # Connect to official Notion MCP server
            session, write, read = await sse_client(NOTION_MCP_URL, headers=headers)
            notion_session = session
            
            # Initialize the session
            await session.initialize()
            notion_mcp_available = True
            logging.info("✅ Official Notion MCP connection established")
            return True
            
        except Exception as e:
            logging.warning(f"Official Notion MCP connection failed: {e}")
    
    logging.info("All MCP connection attempts failed, will use direct Notion API as fallback")
    notion_mcp_available = False
    return False

async def create_notion_task_mcp(task_title: str, task_description: str, slack_channel: str, slack_user: str) -> str:
    """Create a task in Notion via MCP"""
    if not notion_session:
        return None  # Will trigger fallback
    
    try:
        # List available tools to see what we have
        tools_result = await notion_session.list_tools()
        available_tools = [tool.name for tool in tools_result.tools]
        logging.info(f"Available Notion MCP tools: {available_tools}")
        
        # Try to use appropriate tool based on what's available
        task_content = f"""Task detected from Slack

Description: {task_description}

Source: #{slack_channel} (by {slack_user})
Created: {time.strftime('%Y-%m-%d %H:%M:%S')}

Status: Todo"""
        
        # Try different tool names that might be available
        create_result = None
        
        if "create-page" in available_tools:
            create_result = await notion_session.call_tool(
                "create-page", 
                {
                    "title": task_title,
                    "content": task_content
                }
            )
        elif "create_page" in available_tools:
            create_result = await notion_session.call_tool(
                "create_page", 
                {
                    "title": task_title,
                    "content": task_content
                }
            )
        else:
            # Try the first available tool that looks like it creates content
            create_tools = [t for t in available_tools if 'create' in t.lower()]
            if create_tools:
                logging.info(f"Trying tool: {create_tools[0]}")
                create_result = await notion_session.call_tool(
                    create_tools[0], 
                    {
                        "title": task_title,
                        "content": task_content
                    }
                )
        
        if create_result:
            logging.info(f"MCP task creation result: {create_result}")
            return f"✅ Task created in Notion via MCP: {task_title}"
        else:
            logging.warning("No suitable create tool found in MCP")
            return None
        
    except Exception as e:
        logging.error(f"MCP task creation failed: {e}")
        return None  # Will trigger fallback

def run_async_task(coro):
    """Helper to run async tasks in sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return loop.run_until_complete(coro)

def test_notion_connection():
    """Test connection to Notion API"""
    if not NOTION_ACCESS_TOKEN:
        logging.warning("No Notion access token provided")
        return False
        
    try:
        headers = {
            "Authorization": f"Bearer {NOTION_ACCESS_TOKEN}",
            "Content-Type": "application/json",
            "Notion-Version": NOTION_VERSION
        }
        
        # Test with a simple API call
        response = requests.get(f"{NOTION_API_URL}/users/me", headers=headers)
        
        if response.status_code == 200:
            logging.info("Notion API connection established")
            return True
        else:
            logging.error(f"Notion API test failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"Failed to test Notion API: {e}")
        return False

def create_notion_page(task_title: str, task_description: str, slack_channel: str, slack_user: str) -> str:
    """Create a page in Notion using direct API"""
    if not NOTION_ACCESS_TOKEN:
        return "❌ Notion access token not configured"
    
    try:
        headers = {
            "Authorization": f"Bearer {NOTION_ACCESS_TOKEN}",
            "Content-Type": "application/json",
            "Notion-Version": NOTION_VERSION
        }
        
        # Create page content - we'll create it in workspace root
        page_data = {
            "parent": {"type": "workspace", "workspace": True},
            "properties": {
                "title": {
                    "title": [
                        {
                            "text": {
                                "content": task_title
                            }
                        }
                    ]
                }
            },
            "children": [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": f"Task detected from Slack\n\nDescription: {task_description}\n\nSource: #{slack_channel} (by {slack_user})\nCreated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\nStatus: Todo"
                                }
                            }
                        ]
                    }
                }
            ]
        }
        
        response = requests.post(f"{NOTION_API_URL}/pages", headers=headers, json=page_data)
        
        if response.status_code == 200:
            page_data = response.json()
            page_url = page_data.get("url", "")
            return f"✅ Task created in Notion: {task_title}\n{page_url}"
        else:
            logging.error(f"Notion API error: {response.status_code} - {response.text}")
            return f"❌ Failed to create Notion page: {response.text}"
            
    except Exception as e:
        logging.error(f"Failed to create Notion page: {e}")
        return f"❌ Error creating Notion page: {str(e)}"

def create_notion_task(task_title: str, task_description: str, slack_channel: str, slack_user: str) -> str:
    """Create a task in Notion - tries MCP first, falls back to direct API"""
    
    # Try MCP first if available
    if notion_mcp_available and notion_session:
        try:
            result = run_async_task(create_notion_task_mcp(
                task_title, task_description, slack_channel, slack_user
            ))
            if result:  # MCP succeeded
                return result
        except Exception as e:
            logging.warning(f"MCP creation failed, falling back to direct API: {e}")
    
    # Fallback to direct API
    logging.info("Using direct Notion API for task creation")
    return create_notion_page(task_title, task_description, slack_channel, slack_user)

@bolt_app.event("app_mention")
def on_mention(body, say, ack):
    ack()  # respond to Slack within 3 seconds
    logging.info("app_mention event received: %s", {
        "channel": body.get("event", {}).get("channel"),
        "user": body.get("event", {}).get("user"),
    })
    user_text = strip_mention(body["event"]["text"])
    
    # Check for task creation commands
    if user_text.lower().startswith("create task:") or user_text.lower().startswith("task:"):
        task_text = user_text.split(":", 1)[1].strip()
        channel_name = body.get("event", {}).get("channel", "unknown")
        user_id = body.get("event", {}).get("user", "unknown")
        
        try:
            # Generate a task title from the description
            task_title = task_text[:50] + "..." if len(task_text) > 50 else task_text
            
            # Create task in Notion (tries MCP first, falls back to API)
            result = create_notion_task(
                task_title=task_title,
                task_description=task_text,
                slack_channel=channel_name,
                slack_user=user_id
            )
            
            say(result, thread_ts=body["event"]["ts"])
            return
            
        except Exception as e:
            logging.error(f"Error creating Notion task: {e}")
            say(f"❌ Error creating task: {str(e)}", thread_ts=body["event"]["ts"])
            return
    
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
                             "Want me to create it in Notion? Just mention me with:\n"
                             "`@bot task: your task description`")
            
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
    # Try to initialize Notion MCP connection
    if MCP_AVAILABLE:
        try:
            mcp_success = run_async_task(init_notion_mcp())
            if not mcp_success:
                logging.info("MCP connection failed, will use direct API")
        except Exception as e:
            logging.warning(f"Could not initialize Notion MCP: {e}")
    
    # Test direct API connection as fallback
    test_notion_connection()
    
    port = int(os.environ.get("PORT", "3000"))
    flask_app.run(host="0.0.0.0", port=port)