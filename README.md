# Slack Python Bot (Bolt + Flask + OpenAI)

A production-ready Slack bot built with:

- Slack Bolt for Python (events and messaging)
- Flask (small web server with `/slack/events` and `/healthz`)
- OpenAI (responds using Chat Completions)
- Gunicorn (WSGI server for production)

This app replies when the bot is mentioned in a channel and when users DM the bot. It strips the `@mention` from messages and sends the remaining text to OpenAI, then posts the model's reply back to Slack.

Note: If your workspace requires Slack “Data Access” approvals, some APIs may be unavailable until the app is approved. This bot only needs basic messaging scopes by default and works without Data Access APIs.

---

## Architecture Overview

- `app.py` defines:
  - `bolt_app`: Slack Bolt app
  - `flask_app`: Flask app (exported for Gunicorn)
  - Route `POST /slack/events` handled by `SlackRequestHandler` (event subscriptions)
  - Route `GET /healthz` returns `ok` (health checks)
  - Event handlers:
    - `app_mention`: replies to mentions in channels
    - `message` (IM only): replies in direct messages

- Environment variables required (fail-fast on startup if missing):
  - `SLACK_BOT_TOKEN` (starts with `xoxb-`)
  - `SLACK_SIGNING_SECRET`
  - `OPENAI_API_KEY`

- Production server: `Procfile` uses `web: gunicorn app:flask_app`.

---

## Prerequisites

- Python 3.10+ (tested on 3.13)
- A Slack app with a Bot User
- OpenAI API key

---

## Slack App Setup (once per workspace)

1. Create a Slack app (From scratch) and enable a Bot User.
2. Bot Token Scopes (minimum):
   - `app_mentions:read`
   - `chat:write`
   - `im:history`
   - `im:write`
3. Event Subscriptions:
   - Enable and set Request URL to `https://<your-domain>/slack/events`
   - Subscribe to bot events:
     - `app_mention`
     - `message.im`
4. Install the app to the workspace. Copy:
   - Bot User OAuth Token (`xoxb-...`) → `SLACK_BOT_TOKEN`
   - Signing Secret → `SLACK_SIGNING_SECRET`

If your org requires approval for “Data Access” APIs, you can use this bot with basic messaging scopes while approval is pending.

---

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

export SLACK_BOT_TOKEN="xoxb-..."
export SLACK_SIGNING_SECRET="..."
export OPENAI_API_KEY="sk-..."

# Start with Gunicorn (recommended)
python -m gunicorn app:flask_app --bind 127.0.0.1:3000

# Health check
curl -s http://127.0.0.1:3000/healthz
```

Expose your local server to Slack using a tunnel (e.g., `ngrok http 3000`) and set the Event Subscriptions Request URL to `https://<tunnel>/slack/events`.

---

## Deploying to Railway

1. Add the repo as a Railway service.
2. Set these Variables in the service:
   - `SLACK_BOT_TOKEN`
   - `SLACK_SIGNING_SECRET`
   - `OPENAI_API_KEY`
   - `PORT` is provided by Railway automatically; the app reads it.
3. Ensure `Procfile` exists with:
   ```
   web: gunicorn app:flask_app
   ```
4. Deploy. Check Deploy Logs. A successful boot shows Gunicorn starting workers and you can hit `/healthz`.

---

## Troubleshooting

- Missing variables: the app fails fast with a clear message like `Missing required environment variables: SLACK_BOT_TOKEN, ...`.
- Slack invalid_auth: ensure `SLACK_BOT_TOKEN` is a Bot token (`xoxb-...`) and the app is installed in the workspace.
- Gunicorn error “failed to find attribute 'flask_app' in 'app'”: indicates an import-time error. Check logs above for the real exception.
- OpenAI/httpx TypeError about `proxies`: resolved by pinning `httpx==0.27.2` (already in `requirements.txt`).
- Event retries from Slack: handlers `ack()` immediately to avoid timeouts.

---

## How It Works

1. Slack sends events to `/slack/events`.
2. Bolt parses and routes them to handlers.
3. The handler strips mentions, calls OpenAI Chat Completions (`gpt-4o-mini`), and posts the reply.
4. Flask exposes `/healthz` for platform health checks; Gunicorn runs the app in production.

---

## Security Notes

- Keep secrets in environment variables only. Do not commit them.
- Rotate Slack and OpenAI keys periodically.
- Limit scopes to the minimum needed.

---

## License

MIT (or your choice). Update this section as appropriate for your project.


