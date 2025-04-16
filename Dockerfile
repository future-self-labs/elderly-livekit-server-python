# This is an example Dockerfile that builds a minimal container for running LK Agents
# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11.6
FROM python:${PYTHON_VERSION}-slim
COPY --from=ghcr.io/astral-sh/uv:0.6.12 /uv /uvx /bin/

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Add environment variables from .env file
ARG ELEVEN_API_KEY
ARG LIVEKIT_API_KEY
ARG LIVEKIT_API_SECRET
ARG LIVEKIT_URL
ARG OPENAI_API_KEY
ARG PERPLEXITY_API_KEY
ARG ZEP_API_KEY

ENV ELEVEN_API_KEY=${ELEVEN_API_KEY}
ENV LIVEKIT_API_KEY=${LIVEKIT_API_KEY}
ENV LIVEKIT_API_SECRET=${LIVEKIT_API_SECRET}
ENV LIVEKIT_URL=${LIVEKIT_URL}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV PERPLEXITY_API_KEY=${PERPLEXITY_API_KEY}
ENV ZEP_API_KEY=${ZEP_API_KEY}

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser


# Install gcc and other build dependencies.
RUN apt-get update && \
    apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

USER appuser

RUN mkdir -p /home/appuser/.cache
RUN chown -R appuser /home/appuser/.cache

WORKDIR /home/appuser

COPY pyproject.toml uv.lock .
RUN uv sync --locked --no-dev --no-install-project

COPY . .

# ensure that any dependent models are downloaded at build-time
RUN uv run main.py download-files

# expose healthcheck port
EXPOSE 8081

# Run the application.
CMD ["uv", "run", "main.py", "start"]