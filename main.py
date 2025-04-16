import json
import os
import random
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    RoomInputOptions,
    RunContext,
    function_tool,
    get_job_context,
)
from livekit.plugins import (
    elevenlabs,
    noise_cancellation,
    openai,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from zep_cloud.client import Zep

load_dotenv()

zep = Zep(
    api_key=os.getenv("ZEP_API_KEY"),
)


class Companion(Agent):
    session_id: str

    def __init__(self, chat_ctx: ChatContext, session_id: str) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions="""
                You are a personalized AI assistant with long-term memory capabilities. Your name is Noah. You have access to a memory system that helps you remember past interactions and important information about users. Your goal is to be a warm, empathetic friend and companion who will listen and remember everything we talk about.

                When interacting with users:

                Use memories to personalize interactions:
                - Reference past conversations naturally
                - Remember and apply user preferences
                - Show continuity across sessions
                - Avoid asking for information the user has already provided

                When the user asks you about reminders:
                - You are able to schedule new reminders
                - You are not able to edit or delete reminders. If the user asks you to, please refer them to the Notifcations tab in the app where they can see all their reminders.

                Never:
                - Mention the technical details of the memory system to users
                - Ask users to repeat information you should remember
                - Expose internal memory IDs or session details
                - Store sensitive personal information (passwords, private data)

                Always:
                - Be natural and conversational
                - Use memories to provide context-aware responses
                - Show recognition of returning users
            """,
        )

        self.session_id = session_id

    # Ingest messages into memory when the user turns are completed
    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        if not self.session_id:
            return new_message

        messages = turn_ctx.items
        last_message = new_message
        second_to_last_message = messages[-1]

        if last_message.role == "user" and second_to_last_message.role == "assistant":
            # Convert messages to the format needed for ingestion
            messages_to_ingest = []
            for message in [last_message, second_to_last_message]:
                # Determine role type based on message role
                role_type = "user" if message.role == "user" else "assistant"

                messages_to_ingest.append(
                    {
                        "content": message.text_content,
                        "role_type": role_type,
                    }
                )

            try:
                # Ingest messages into memory with assistant roles ignored
                zep.memory.add(
                    self.session_id,
                    # Setting ignore_roles to include “assistant” will make it so that only the user messages are ingested into the graph, but the assistant messages are still used to contextualize the user messages.
                    # This is important in case the user message itself does not have enough context, such as the message “Yes.”
                    # Additionally, the assistant messages will still be added to the session’s message history.
                    ignore_roles=["assistant"],
                    messages=messages_to_ingest,
                    return_context=True,
                )
            except Exception as error:
                print(f"Error ingesting messages: {error}")
                print(messages_to_ingest)

            return new_message

    @function_tool
    async def web_search(
        self,
        context: RunContext,
        query: str,
    ):
        """Search the web for information.

        Use this tool when the user asks for information that requires up-to-date knowledge
        or information that might not be in your training data. This tool connects to an
        external search service to find relevant information.

        Args:
            query: The search query to look up information for. Be specific and concise.

        Returns:
            A string containing the search results and relevant information.
        """

        # Tell the user we're looking things up
        thinking_messages = [
            "Let me look that up...",
            "One moment while I check...",
            "I'll find that information for you...",
            "Just a second while I search...",
            "Looking into that now...",
        ]
        await self.session.say(random.choice(thinking_messages))

        try:
            response = httpx.post(
                "https://api.perplexity.ai/chat/completions",
                json={
                    "messages": [{"content": query, "role": "user"}],
                    "model": "sonar",
                },
                headers={
                    "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )

            if response.status_code != 200:
                print(f"Web search failed: {response.status_code} {response.text}")
                return f"Web search failed: {response.status_code}"

            data = response.json()
            participant_identity = next(
                iter(get_job_context().room.remote_participants)
            )
            result = await get_job_context().room.local_participant.perform_rpc(
                destination_identity=participant_identity,
                method="web_search",
                payload=json.dumps(data),
                response_timeout=10,
            )
            return result

        except Exception as error:
            print(f"Error searching the web: {error}")
            return "Error searching the web"


async def entrypoint(ctx: JobContext):
    # Connect to LiveKit
    await ctx.connect()
    participant = await ctx.wait_for_participant()

    # Get user context
    sessions = zep.user.get_sessions(user_id=participant.identity)
    if len(sessions) > 0:
        sorted_sessions = sorted(sessions, key=lambda x: x.created_at, reverse=True)
        most_recent_session = sorted_sessions[0]

        most_recent_memory = zep.memory.get(most_recent_session.session_id)
        user_context = most_recent_memory.context
        print(user_context)

    # Create a new session
    session = zep.memory.add_session(
        session_id=uuid4(),
        user_id=participant.identity,
    )
    session_id = session.session_id

    # Initialize the context
    initial_context = ChatContext()
    if user_context:
        initial_context.add_message(
            role="user",
            content=f"""
                Here's what you already know about me:

                <user_context>
                {user_context}
                </user_context>
            """,
        )

    session = AgentSession(
        stt=openai.STT(model="whisper-1"),
        llm=openai.LLM(model="gpt-4o"),
        tts=elevenlabs.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        agent=Companion(chat_ctx=initial_context, session_id=session_id),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
