# CLAUDE.md — ai_model_monitor

## Project Purpose

This project monitors AI model releases and leaderboard-relevant news to support **Kalshi prediction market trading** on KXTOPMODEL and KXLLM1 contracts. These contracts settle based on the [Chatbot Arena](https://arena.ai/leaderboard/text/overall-no-style-control) text leaderboard (no-style-control).

The only notifications that matter are ones that could move Kalshi AI leaderboard contract prices. Everything else is noise.

## Architecture

Single-file Python script (`ai_news_monitor.py`) using a **poll → diff → notify** pattern:
1. Fetch data from RSS feeds, Hugging Face API, GitHub Releases API, and Hacker News Algolia API
2. Diff against cached state (JSON file persisted via GitHub Actions cache)
3. Send Discord webhook embeds for new items
4. Runs every 30 minutes via GitHub Actions cron, with 4 checks per run at randomized 60–120s intervals

## The Core Problem to Solve

**Hacker News notifications are overwhelmingly irrelevant.** The current keyword-based HN search via Algolia returns too many false positives — generic AI industry posts, tooling announcements, opinion pieces, and "Show HN" projects that have zero bearing on which model will top the Arena leaderboard.

Examples of **irrelevant** HN posts that should NOT trigger notifications:
- "LogClaw – Open-source AI SRE that auto-creates tickets from logs" (devops tooling)
- "Should Sam Altman fear token compression?" (opinion/speculation)
- "IonRouter (YC W26) – High-throughput, low-cost inference" (inference infrastructure)
- "Every Developer in the World, Ranked" (completely unrelated)
- "Slop or not – can you tell AI writing from human?" (AI detection, not model news)
- General AI ethics, regulation, funding, hiring, or business strategy posts
- AI application/product launches that aren't new foundation models
- Infrastructure, tooling, or wrapper projects (RAG frameworks, inference servers, AI agents, etc.)

Examples of **relevant** HN posts that SHOULD trigger notifications:
- "Anthropic releases Claude 4" (new model from tracked company)
- "GPT-5 scores first on Chatbot Arena" (direct leaderboard movement)
- "DeepSeek-V4 released on Hugging Face" (new model upload from tracked org)
- "Qwen3 tops LMSYS Arena" (leaderboard result)
- "Meta releases Llama 4 405B" (new model from tracked company)
- "Google announces Gemini 2.5 Pro" (new model from tracked company)
- "Mistral Large 3 benchmark results" (benchmark performance of tracked model)
- Posts linking to official blog posts or release announcements from tracked companies

## Tracked Companies & Organizations

The only companies/orgs that matter for Kalshi leaderboard contracts:
- **Anthropic** (Claude)
- **OpenAI** (GPT, o-series)
- **Google DeepMind** (Gemini)
- **Meta AI** (Llama)
- **xAI** (Grok)
- **Mistral AI** (Mistral, Mixtral)
- **DeepSeek** (DeepSeek)
- **Qwen / Alibaba** (Qwen)
- **Amazon** (Nova)

## Guiding Principles for Filtering Improvements

When modifying the Hacker News monitoring logic:

1. **Relevance = could this change Kalshi leaderboard contract prices?** If a post doesn't signal a new model release, a benchmark result, or a leaderboard position change from one of the tracked companies, it's noise.

2. **Prefer precision over recall.** Missing a borderline HN post is acceptable because the RSS feeds and Hugging Face/GitHub monitors already cover official announcements. HN is a supplementary signal — it should only fire for posts that the other sources might miss or where HN discussion adds early signal.

3. **Filter aggressively on HN results.** Consider both keyword refinement (tighter search queries) and post-fetch filtering (title/content analysis to reject false positives). A two-stage approach — broad fetch, then strict filter — is better than trying to craft a perfect Algolia query.

4. **Title-based heuristics to reject noise:**
   - "Show HN:" posts are almost never relevant unless they literally announce a new model from a tracked company
   - Posts about AI tooling, frameworks, wrappers, agents, RAG, fine-tuning tutorials, etc. are noise
   - Posts about AI regulation, ethics, business strategy, funding rounds, hiring are noise
   - Posts about AI applications (image gen, code assistants, chatbot products) are noise unless they reference a new foundation model
   - Posts containing "ranked", "developer", "startup", "YC", "launch", "hiring", "raises", "funding" without also containing a tracked company/model name are almost certainly noise

5. **Positive signal keywords** (should appear alongside a tracked company/model name):
   - "release", "released", "launches", "announces", "new model"
   - "benchmark", "leaderboard", "arena", "LMSYS", "Chatbot Arena"
   - "scores", "tops", "beats", "surpasses", "#1"
   - Specific model family names: Claude, GPT, Gemini, Llama, Grok, Mistral, Mixtral, DeepSeek, Qwen, Nova

6. **Points/comment thresholds** can help but aren't sufficient alone — a high-point post about "AI bubble" discourse is still noise.

## Code Conventions

- Single-file architecture — keep everything in `ai_news_monitor.py`
- State is a flat JSON file; keep the schema minimal
- Discord embeds use color-coding per company — maintain this pattern
- Use `logging` module throughout; no print statements
- The script must remain runnable both locally (CLI) and in GitHub Actions
- Dependencies should stay minimal (check `requirements.txt` before adding anything)
- All secrets come from environment variables or GitHub repository secrets

## Testing Changes

- Use `--dry-run` to verify filtering changes without posting to Discord
- Use `--seed` on first run to avoid notification flood
- When modifying HN filtering, test with real Algolia API responses to confirm false positives are rejected and true positives pass through
- Consider logging rejected HN posts at DEBUG level so filtering effectiveness can be audited

## Files

- `ai_news_monitor.py` — all monitoring logic (fetch, diff, notify)
- `.github/workflows/ai-news-monitor.yml` — cron schedule and GitHub Actions config
- `requirements.txt` — Python dependencies
- `.gitignore` — standard ignores
- `README.md` — user-facing documentation

## Related Repositories

- [kalshi_top_model_events](https://github.com/acarter881/kalshi_top_model_events) — Kalshi price tracking for AI leaderboard contracts
- [LLM-Leaderboard](https://github.com/acarter881/LLM-Leaderboard) — Arena leaderboard scraper with 30-second polling and Discord alerts
