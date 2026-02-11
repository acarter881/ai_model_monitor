# ai_model_monitor

Send a message in a Discord server when relevant information related to a new AI model is released.

Monitors companies that have historically competed for #1 on the [Chatbot Arena text leaderboard](https://lmarena.ai/leaderboard/text/overall) and sends Discord notifications when new blog posts, model uploads, or GitHub releases are detected.

Built for use with [Kalshi](https://kalshi.com) prediction markets — get early signals about model releases so you can place bets before the market adjusts.

## Sources Monitored

| Source Type | Description |
|---|---|
| **Blog RSS/Atom feeds** | Official company blogs (Anthropic, OpenAI, Google DeepMind, Meta AI, xAI, Mistral, Qwen) |
| **Hugging Face API** | New model uploads from key organizations (meta-llama, mistralai, deepseek-ai, Qwen, google, xai-org, amazon) |
| **GitHub Releases** | New releases from model repos (meta-llama/llama-models, deepseek-ai/DeepSeek-V3, etc.) |
| **Hacker News** | Keyword-based search via Algolia API for AI model news and leaderboard updates |

## Companies Tracked

Anthropic, OpenAI, Google DeepMind, Meta AI, xAI, Mistral AI, DeepSeek, Qwen (Alibaba), Amazon

## Setup

### 1. Repository Secret

Add your Discord webhook URL as a repository secret:

- **Name:** `DISCORD_WEBHOOK_URL`
- **Value:** `https://discord.com/api/webhooks/...`

The `GITHUB_TOKEN` is provided automatically by GitHub Actions for higher API rate limits.

### 2. Enable the Workflow

The GitHub Actions workflow runs on a cron schedule (every 10 minutes). Each run performs 4 checks with randomized 60–120 second intervals between them, giving roughly one check every ~1.5 minutes spread across the 10-minute window.

### 3. First Run (Seeding)

The first run automatically seeds the state file without sending notifications. This avoids flooding Discord with all existing blog posts and models. You can also trigger a manual seed run:

```
workflow_dispatch → seed: true
```

## Local Usage

```bash
pip install -r requirements.txt

# Seed initial state (no notifications)
python ai_news_monitor.py --seed --state-file state.json

# Single check
python ai_news_monitor.py --webhook-url "https://discord.com/api/webhooks/..." --state-file state.json

# Dry run (log only)
python ai_news_monitor.py --dry-run --state-file state.json

# Loop mode
python ai_news_monitor.py --loop --max-checks 4 --min-interval-seconds 60 --max-interval-seconds 120 --state-file state.json
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--webhook-url` | `$DISCORD_WEBHOOK_URL` | Discord webhook URL |
| `--state-file` | `.github/state/ai_news_state.json` | Path to JSON state file |
| `--force-send` | `false` | Send a status embed even when nothing changed |
| `--dry-run` | `false` | Log what would be sent without posting to Discord |
| `--seed` | `false` | Build state without sending notifications |
| `--loop` | `false` | Run multiple checks in a loop |
| `--max-checks` | `4` | Max iterations in loop mode |
| `--min-interval-seconds` | `60` | Min sleep between loop iterations |
| `--max-interval-seconds` | `120` | Max sleep between loop iterations |

## Architecture

Follows the same **poll → diff → notify** pattern as [kalshi_top_model_events](https://github.com/acarter881/kalshi_top_model_events):

1. **Fetch** data from all sources (RSS feeds, HF API, GH API, HN API)
2. **Diff** the new snapshot against the cached state
3. **Notify** via Discord webhook with color-coded embeds per company
4. **Persist** the new state via GitHub Actions cache

State is stored as a JSON file and cached between workflow runs using `actions/cache`.
