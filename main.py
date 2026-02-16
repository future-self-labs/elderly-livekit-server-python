import asyncio
import json
import os
import traceback
from functools import partial
from urllib.parse import quote
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
    silero,
    turn_detector,
)
from openai.types.beta.realtime.session import (
    InputAudioTranscription,
    TurnDetection,
)
from zep_cloud.client import Zep

from agents.companion_agent import CompanionAgent
from agents.onboarding_agent import OnboardingAgent
from prompts import load_all_skills

load_dotenv()

# Patch av 13 flag names to match what livekit-agents expects (av 14 API)
# av 13 uses UPPERCASE (NOBUFFER, FLUSH_PACKETS), livekit-agents expects snake_case (no_buffer, flush_packets)
import av.container
_Flags = av.container.Flags
if not hasattr(_Flags, "no_buffer") and hasattr(_Flags, "NOBUFFER"):
    _Flags.no_buffer = _Flags.NOBUFFER
    _Flags.flush_packets = _Flags.FLUSH_PACKETS

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


async def _create_zep_session(user_id: str) -> str | None:
    """Create a new Zep session, running sync call in thread executor."""
    try:
        session = await _run_sync(
            zep.memory.add_session,
            session_id=uuid4(),
            user_id=user_id,
        )
        return session.session_id
    except Exception as e:
        print(f"[Zep] Error creating session (non-fatal): {e}")
        return None


async def _get_people(user_id: str) -> list:
    """Fetch the elderly user's people network from the API."""
    try:
        data = await get_api_data(f"/people/{user_id}")
        return data.get("people", [])
    except Exception as e:
        print(f"[Memory] Error fetching people: {e}")
        return []


async def _get_upcoming_events(user_id: str, days: int = 7) -> list:
    """Fetch upcoming events for the elderly user."""
    try:
        data = await get_api_data(f"/events/{user_id}/upcoming?days={days}")
        return data.get("events", [])
    except Exception as e:
        print(f"[Memory] Error fetching events: {e}")
        return []


async def _log_wellbeing(user_id: str, mood_score: int | None = None,
                          conversation_minutes: int = 0, topics: list | None = None,
                          concerns: list | None = None):
    """Log a wellbeing entry after a conversation."""
    from datetime import date
    try:
        await get_api_data(
            "/wellbeing",
            method="POST",
            json={
                "elderlyUserId": user_id,
                "date": date.today().isoformat(),
                "moodScore": mood_score,
                "conversationMinutes": conversation_minutes,
                "topics": topics or [],
                "concerns": concerns or [],
            },
        )
        print(f"[Wellbeing] Logged for {user_id}")
    except Exception as e:
        print(f"[Wellbeing] Error logging: {e}")


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
        room_name = ctx.room.name or ""

        # Outbound calls have room name "call-{userId}" — extract userId directly
        if room_name.startswith("call-"):
            extracted_id = room_name[5:]
            print(f"[Agent] Outbound call detected — userId from room: {extracted_id}")
            try:
                user = await get_api_data(f"/users/{extracted_id}")
                user_id = user["id"]
                elderly_user = user
            except Exception as e:
                print(f"[Agent] User lookup by room ID failed: {e}")
                user = {"name": "Caller", "id": extracted_id}
                elderly_user = user
        else:
            # Inbound call — look up by phone number
            try:
                user = await get_api_data(f"/users/search?phoneNumber={quote(phone_number, safe='')}")

                if user.get("type") == "family_member":
                    is_family_member = True
                    user_id = user["userId"]
                    elderly_user = await get_api_data(f"/users/{user_id}")
                else:
                    user_id = user["id"]
                    elderly_user = user
            except Exception as e:
                print(f"[Agent] SIP caller lookup failed (proceeding as unknown): {e}")
                user = {"name": "Caller", "id": user_id}
                elderly_user = user
    else:
        user = await get_api_data(f"/users/{user_id}")
        elderly_user = user

    # Parallelize: fetch Zep context, create Zep session, load people + events
    if not is_family_member:
        zep_context_task = asyncio.create_task(_get_zep_context(user_id))
        zep_session_task = asyncio.create_task(_create_zep_session(user_id))
        people_task = asyncio.create_task(_get_people(user_id))
        events_task = asyncio.create_task(_get_upcoming_events(user_id))

        user_context, session_id, people_data, upcoming_events = await asyncio.gather(
            zep_context_task, zep_session_task, people_task, events_task
        )

        if user_context:
            print(f"[Zep] Loaded context ({len(user_context)} chars)")
        if people_data:
            print(f"[Memory] Loaded {len(people_data)} people")
        if upcoming_events:
            print(f"[Memory] Loaded {len(upcoming_events)} upcoming events")
    else:
        session_id = await _create_zep_session(user_id)
        people_data = []
        upcoming_events = []

    # Build initial context with skills
    # NOTE: context messages use XML tags (language-neutral) with minimal English scaffolding
    # to avoid biasing the model toward English. The system prompt enforces the user's language.
    initial_context = ChatContext()
    initial_context.add_message(
        role="assistant",
        content=f"""<skills>
{_SKILLS_CONTEXT}
</skills>""",
    )

    if user_context:
        initial_context.add_message(
            role="assistant",
            content=f"""<user_context>
{user_context}
</user_context>""",
        )

    # Inject people from the memory vault
    if people_data:
        people_text = "\n".join(
            f"- {p['name']} ({p['relationship']})"
            + (f", nickname: {p['nickname']}" if p.get("nickname") else "")
            + (f", birthday: {p['birthDate']}" if p.get("birthDate") else "")
            + (f" — {p['notes']}" if p.get("notes") else "")
            for p in people_data
        )
        initial_context.add_message(
            role="assistant",
            content=f"""<people>
{people_text}
</people>""",
        )

    # Inject upcoming events — only mention naturally, don't lead with them
    if upcoming_events:
        events_text = "\n".join(
            f"- {e['title']} ({e['type']}) — {e['date']}"
            for e in upcoming_events
        )
        initial_context.add_message(
            role="assistant",
            content=f"""<upcoming_events>
{events_text}
</upcoming_events>""",
        )

    if attributes.get("initialRequest"):
        initial_context.add_message(
            role="user",
            content=f"""<user_request>
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

    return agent, user


# ---------------------------------------------------------------------------
# Single entrypoint — routes between Realtime and Pipeline based on
# dispatch metadata. The server sets metadata="pipeline" for pipeline tokens.
# ---------------------------------------------------------------------------

DEFAULT_VOICE_ID = "bIHbv24MWmeRgasZH58o"  # ElevenLabs default (Will)

async def entrypoint(ctx: JobContext):
    print(f"[Agent] entrypoint called — metadata={ctx.job.metadata!r}")

    try:
        agent, user_data = await _build_context_and_agent(ctx)
    except Exception as e:
        print(f"[Agent] FATAL: _build_context_and_agent failed: {e}")
        traceback.print_exc()
        return

    user_language = (user_data or {}).get("language", "nl")
    user_id_log = (user_data or {}).get("id", "unknown")
    print(f"[Agent] user_language={user_language}, user_id={user_id_log}")

    # Parse metadata — can be plain "pipeline" string or JSON {"mode":"pipeline","voiceId":"..."}
    raw_metadata = (ctx.job.metadata or "").strip()
    use_pipeline = False
    voice_id = DEFAULT_VOICE_ID

    if raw_metadata == "pipeline":
        use_pipeline = True
    elif raw_metadata.startswith("{"):
        try:
            meta = json.loads(raw_metadata)
            use_pipeline = meta.get("mode") == "pipeline"
            voice_id = meta.get("voiceId") or DEFAULT_VOICE_ID
        except Exception:
            pass

    if use_pipeline:
        print(f"[Agent] Using PIPELINE mode (Deepgram + GPT-4o-mini + ElevenLabs, voice={voice_id})")
        # Verify required API keys are present
        if not os.getenv("DEEPGRAM_API_KEY"):
            print("[Agent] ERROR: DEEPGRAM_API_KEY is not set — Pipeline mode cannot work")
        if not os.getenv("ELEVEN_API_KEY"):
            print("[Agent] ERROR: ELEVEN_API_KEY is not set — Pipeline mode cannot work")
        try:
            session = AgentSession(
                turn_detection="vad",
                vad=silero.VAD.load(
                    min_silence_duration=0.5,
                ),
                stt=deepgram.STT(
                    model="nova-2",
                    language=user_language,
                ),
                llm=openai.LLM(
                    model="gpt-4o-mini",
                    temperature=0.8,
                ),
                tts=elevenlabs.TTS(
                    model="eleven_turbo_v2_5",
                    voice_id=voice_id,
                    language=user_language,
                    voice_settings=elevenlabs.VoiceSettings(
                        stability=0.35,
                        similarity_boost=0.8,
                        style=0.4,
                    ),
                ),
                allow_interruptions=True,
                min_interruption_duration=0.8,
                min_endpointing_delay=0.5,
            )
            print("[Agent] Pipeline AgentSession created successfully")
        except Exception as e:
            print(f"[Agent] ERROR creating Pipeline session: {e}")
            traceback.print_exc()
            return
    else:
        print("[Agent] Using REALTIME mode (OpenAI Realtime API)")
        try:
            session = AgentSession(
                allow_interruptions=True,
                min_interruption_duration=0.6,
                llm=openai.realtime.RealtimeModel(
                    voice="ash",
                    turn_detection=TurnDetection(
                        type="server_vad",
                        threshold=0.6,
                        prefix_padding_ms=300,
                        silence_duration_ms=500,
                    ),
                    input_audio_transcription=InputAudioTranscription(
                        model="whisper-1",
                        language=user_language,
                    ),
                ),
            )
            print("[Agent] Realtime AgentSession created successfully")
        except Exception as e:
            print(f"[Agent] ERROR creating Realtime session: {e}")
            traceback.print_exc()
            return

    try:
        await session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        print("[Agent] session.start() completed")
    except Exception as e:
        print(f"[Agent] ERROR during session.start(): {e}")
        traceback.print_exc()
        return

    # Build greeting instruction in the user's language
    lang_name = {"nl": "Dutch", "en": "English", "de": "German", "fr": "French", "es": "Spanish", "tr": "Turkish"}.get(user_language, "Dutch")
    greeting_instruction = f"Greet the user warmly in {lang_name}. Keep it short and natural — like a friend saying hello."

    try:
        await session.generate_reply(
            instructions=greeting_instruction,
        )
        print("[Agent] generate_reply() completed")
    except Exception as e:
        print(f"[Agent] ERROR during generate_reply(): {e}")
        traceback.print_exc()


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            agent_name="noah",
            entrypoint_fnc=entrypoint,
        )
    )
