# LiveKit Server

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
