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

### uv sync

This will install the dependencies and create a lockfile if one doesn't exist.

```bash
uv sync
```

### Download the required models

```bash
uv run main.py download-files
```

### uv run

This will run the script in dev mode for local development.

```bash
uv run main.py dev
```
