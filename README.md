# LiveKit Server

This is the LiveKit server for the elderly companion app. It requires that the elderly-api is running.

Livekit exposes two agents: a family onboarding agent and a companion agent.

The family onboarding agent is responsible for handling the onboarding process for a new family member. It is responsible for gathering information about the family member and their relationship to the elderly user.

The companion agent is responsible for handling the conversation with the elderly user. It is responsible for responding to the user's messages and is the main assistant.

## Setup

### uv

This project uses uv to manage dependencies.

```bash
brew install uv
```

or

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### uv sync

This will install the dependencies and create a lockfile if one doesn't exist.

```bash
uv sync
```

### Download the required Language Models

```bash
uv run main.py download-files
```

### uv run

This will run the script in dev mode for local development.

```bash
uv run main.py dev
```

## Deployment

This project is deployed as a Docker container on Render.com (see `Dockerfile`). It is automatically built & deployed when changes are pushed to the `main` branch.

### 3rd-party services

#### Zep

[Zep](https://getzep.com) is a memory layer for LLMs. It is used to store and retrieve user facts. We use the hosted service on the free tier.

### N8N

[N8N](https://n8n.io) is a workflow automation tool. It is used to faciliate callback requests (i.e when the user requests the AI to call them back at a certain time).

### Perplexity

[Perplexity](https://www.perplexity.ai) is a search engine for LLMs. We use it to look up information on the internet when the user asks.
