import os
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
    noise_cancellation,
    openai,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from zep_cloud.client import Zep

from agents.companion_agent import CompanionAgent
from agents.onboarding_agent import OnboardingAgent

load_dotenv()


async def get_api_data(path: str, **kwargs) -> dict:
    """
    Fetch data from the API.

    Args:
        path: The API path to fetch from (should start with '/')
        **kwargs: Additional arguments to pass to httpx.AsyncClient.request

    Returns:
        The JSON response as a dictionary
    """
    api_url = os.getenv("API_URL")
    if not api_url:
        raise ValueError("API_URL environment variable is not set")
    else:
        print(f"API_URL: {api_url}")

    url = f"{api_url}{path}"

    headers = kwargs.pop("headers", {})
    headers["Content-Type"] = "application/json"

    async with httpx.AsyncClient() as client:
        response = await client.request(
            method=kwargs.pop("method", "GET"), url=url, headers=headers, **kwargs
        )
        response.raise_for_status()
        return response.json()


zep = Zep(
    api_key=os.getenv("ZEP_API_KEY"),
)


async def entrypoint(ctx: JobContext):
    # Connect to LiveKit
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    attributes = participant.attributes
    user_id = participant.identity

    is_family_member = False
    # will hold the user context from the memory store (Zep), i.e the facts about the user that have been gathered throughout the conversations
    user_context = None
    # the user that is connecting to the agent
    user = None
    # if it's a family member, we also need to store the elderly user's name and id
    elderly_user = None

    # fetch the user from the API
    # if the user is calling from a phone number, lookup the user by phone number
    if participant.identity.startswith("sip_"):
        phone_number = participant.identity[4:]
        user = await get_api_data(f"/users/search?phoneNumber={phone_number}")

        if user["type"] == "family_member":
            is_family_member = True
            user_id = user["userId"]
            elderly_user = await get_api_data(f"/users/{user_id}")
        else:
            user_id = user["id"]

    # if a user is not calling from a phone number, they are using the app, and so we know they are the elderly user
    else:
        user = await get_api_data(f"/users/{user_id}")
        elderly_user = user

    # if it is the main user, retrieve all the facts from the memory store (Zep)
    if not is_family_member:
        # Get user context
        # get sessions for the family "owner" user ID
        sessions = zep.user.get_sessions(user_id)

        if len(sessions) > 0:
            sorted_sessions = sorted(sessions, key=lambda x: x.created_at, reverse=True)
            most_recent_session = sorted_sessions[0]

            most_recent_memory = zep.memory.get(most_recent_session.session_id)

            user_context = most_recent_memory.context
            print(user_context)

    # Create a new session in the memory store (Zep)
    session = zep.memory.add_session(
        session_id=uuid4(),
        user_id=user_id,
    )

    session_id = session.session_id

    # Initialize the context with the facts from the memory store (Zep)
    initial_context = ChatContext()
    if user_context:
        initial_context.add_message(
            role="user",
            content=f"""
                Here's what you already know about me. These are facts gathered throughout our conversations and also facts contributed by family members in other conversations.

                <user_context>
                {user_context}
                </user_context>
            """,
        )

    # Add the initial request from the user if the Agent is calling the user (triggered through N8N as a result of the scheduled workflow - see tool call in the companion agent)
    if attributes.get("initialRequest"):
        initial_context.add_message(
            role="user",
            content=f"""
                You, the Companion, are currently in a phone call with me, the user. Some time ago, I asked you to discuss a topic with you. You have now been connected with  me via a phone call to discuss this topic. 
                
                Here's what the topic I want to discuss with you:

                <user_request>
                {attributes["initialRequest"]}
                </user_request>
            """,
        )

    session = AgentSession(
        allow_interruptions=True,
        turn_detection=MultilingualModel(),
        # we disable realtime turn detection because we need the end_user_turn hook to be called in the Agent
        llm=openai.realtime.RealtimeModel(
            voice="ash", turn_detection=None, input_audio_transcription=None
        ),
        stt=openai.STT(model="whisper-1", language="nl"),
        vad=ctx.proc.userdata["vad"],
    )

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


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            agent_name="noah",
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
