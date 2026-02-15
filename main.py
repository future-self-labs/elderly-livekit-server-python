import asyncio
import os
from functools import partial
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    RoomInputOptions,
)
from livekit.plugins import (
    deepgram,
    elevenlabs,
    noise_cancellation,
    openai,
)
from zep_cloud.client import Zep

from agents.companion_agent import CompanionAgent
from agents.onboarding_agent import OnboardingAgent
from prompts import load_all_skills

load_dotenv()

# Load skills once at startup (not per-session)
_SKILLS_CONTEXT = load_all_skills()

# Shared HTTP client for connection pooling (reused across requests)
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=15.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
    return _http_client


async def get_api_data(path: str, **kwargs) -> dict:
    """Fetch data from the API using the shared HTTP client."""
    api_url = os.getenv("API_URL")
    if not api_url:
        raise ValueError("API_URL environment variable is not set")

    url = f"{api_url}{path}"
    headers = kwargs.pop("headers", {})
    headers["Content-Type"] = "application/json"

    client = get_http_client()
    response = await client.request(
        method=kwargs.pop("method", "GET"), url=url, headers=headers, **kwargs
    )
    response.raise_for_status()
    return response.json()


zep = Zep(
    api_key=os.getenv("ZEP_API_KEY"),
)


async def _run_sync(func, *args, **kwargs):
    """Run a blocking/sync function in a thread executor to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


async def _get_zep_context(user_id: str) -> str | None:
    """Fetch user context from Zep memory, running sync calls in thread executor."""
    try:
        sessions = await _run_sync(zep.user.get_sessions, user_id)

        if len(sessions) > 0:
            sorted_sessions = sorted(sessions, key=lambda x: x.created_at, reverse=True)
            most_recent_session = sorted_sessions[0]
            most_recent_memory = await _run_sync(zep.memory.get, most_recent_session.session_id)
            return most_recent_memory.context
    except Exception as e:
        print(f"[Zep] Error fetching context: {e}")
    return None


async def _create_zep_session(user_id: str) -> str:
    """Create a new Zep session, running sync call in thread executor."""
    session = await _run_sync(
        zep.memory.add_session,
        session_id=uuid4(),
        user_id=user_id,
    )
    return session.session_id


async def _build_context_and_agent(ctx: JobContext):
    """Shared setup for both Realtime and Pipeline entrypoints.

    Connects to LiveKit, identifies the participant, loads Zep context,
    builds ChatContext with skills, and returns the agent + context needed
    to start a session.
    """
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    attributes = participant.attributes
    user_id = participant.identity

    is_family_member = False
    user_context = None
    user = None
    elderly_user = None

    # Fetch user from API
    if participant.identity.startswith("sip_"):
        phone_number = participant.identity[4:]
        user = await get_api_data(f"/users/search?phoneNumber={phone_number}")

        if user["type"] == "family_member":
            is_family_member = True
            user_id = user["userId"]
            elderly_user = await get_api_data(f"/users/{user_id}")
        else:
            user_id = user["id"]
    else:
        user = await get_api_data(f"/users/{user_id}")
        elderly_user = user

    # Parallelize: fetch Zep context + create Zep session at the same time
    if not is_family_member:
        zep_context_task = asyncio.create_task(_get_zep_context(user_id))
        zep_session_task = asyncio.create_task(_create_zep_session(user_id))

        user_context, session_id = await asyncio.gather(zep_context_task, zep_session_task)

        if user_context:
            print(f"[Zep] Loaded context ({len(user_context)} chars)")
    else:
        session_id = await _create_zep_session(user_id)

    # Build initial context with skills
    initial_context = ChatContext()
    initial_context.add_message(
        role="assistant",
        content=f"""I have the following skills and capabilities that I can use during our conversation:

{_SKILLS_CONTEXT}""",
    )

    if user_context:
        initial_context.add_message(
            role="user",
            content=f"""Here's what you already know about me from previous conversations and family input:

<user_context>
{user_context}
</user_context>""",
        )

    if attributes.get("initialRequest"):
        initial_context.add_message(
            role="user",
            content=f"""You are calling me to discuss a topic I previously requested. Here's what I want to discuss:

<user_request>
{attributes["initialRequest"]}
</user_request>""",
        )

    # Build the agent
    agent = None
    if is_family_member:
        agent = OnboardingAgent(
            chat_ctx=initial_context,
            session_id=session_id,
            user=user,
            elderly_name=elderly_user["name"],
        )
    else:
        agent = CompanionAgent(
            chat_ctx=initial_context, session_id=session_id, user=user
        )

    return agent


# ---------------------------------------------------------------------------
# Entrypoint 1: OpenAI Realtime API (existing — agent name "noah")
# ---------------------------------------------------------------------------

async def entrypoint(ctx: JobContext):
    agent = await _build_context_and_agent(ctx)

    from livekit.plugins.openai.realtime.realtime_model import TurnDetection, InputAudioTranscription

    session = AgentSession(
        allow_interruptions=True,
        llm=openai.realtime.RealtimeModel(
            voice="ash",
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=200,
                silence_duration_ms=350,
            ),
            input_audio_transcription=InputAudioTranscription(
                model="whisper-1",
                language="nl",
            ),
        ),
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


# ---------------------------------------------------------------------------
# Entrypoint 2: Pipeline STT→LLM→TTS (new — agent name "noah-pipeline")
# Deepgram Nova-2 (streaming STT) + GPT-4o-mini + ElevenLabs multilingual v2
# ---------------------------------------------------------------------------

async def pipeline_entrypoint(ctx: JobContext):
    agent = await _build_context_and_agent(ctx)

    session = AgentSession(
        stt=deepgram.STT(
            model="nova-2",
            language="nl",
        ),
        llm=openai.LLM(
            model="gpt-4o-mini",
        ),
        tts=elevenlabs.TTS(
            model="eleven_multilingual_v2",
            # Default voice — change to a specific voice_id if desired
        ),
        allow_interruptions=True,
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            agent_name="noah",
            entrypoint_fnc=entrypoint,
        ),
        agents.WorkerOptions(
            agent_name="noah-pipeline",
            entrypoint_fnc=pipeline_entrypoint,
        ),
    )
