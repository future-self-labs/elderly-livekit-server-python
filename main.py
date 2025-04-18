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
    ToolError,
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
from lib.n8n import create_scheduled_workflow, delete_scheduled_workflow, get_user_workflows

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
                - You are able to delete existing reminders, but not edit existing ones. 

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
                    # Setting ignore_roles to include "assistant" will make it so that only the user messages are ingested into the graph, but the assistant messages are still used to contextualize the user messages.
                    # This is important in case the user message itself does not have enough context, such as the message "Yes."
                    # Additionally, the assistant messages will still be added to the session's message history.
                    ignore_roles=["assistant"],
                    messages=messages_to_ingest,
                    return_context=True,
                )
            except Exception as error:
                print(f"Error ingesting messages: {error}")
                print(messages_to_ingest)

            return new_message
            
    @function_tool()
    async def get_phone_number(
        context: RunContext,
    ):
        """Retrieve the user's phone number.
        
        Returns:
            A string containing the user's phone number
        """
        try:
            participant_identity = next(iter(get_job_context().room.remote_participants))

            response = await context.session.room.local_participant.perform_rpc(
                destination_identity=participant_identity,
                method="get_phone_number",
            )
            return response
        except Exception:
            raise ToolError("Unable to retrieve user phone number")

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
                timeout=25.0,
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
                response_timeout=25,
            )
            return result

        except Exception as error:
            print(f"Error searching the web: {error}")
            return "Error searching the web"

    @function_tool
    async def schedule_task(
        self,
        context: RunContext,
        phone_number: str,
        cron_expression: str,
        message: str,
        title: str,
    ):
        """Schedule a task to be executed at a specific time.

        This tool creates and activates a workflow in n8n that will trigger a task to be executed at the specified time.

        Args:
            phone_number: The phone number to call. If not known, can be retrieved using the get_phone_number tool.
            cron_expression: The cron expression specifying when to trigger the task.
                message: The topic or message that the user wants to discuss.
            title: The title of the task.
            message: The message to send to the user.
        Returns:
            A string indicating whether the task was successfully scheduled.
        """

        try:
            # Get the user ID from the context
            participant_identity = next(
                iter(get_job_context().room.remote_participants)
            )

            # Create and activate the workflow in n8n
            await create_scheduled_workflow(
                cron=cron_expression,
                phone_number=phone_number,
                user_id=participant_identity,
                message=message,
                title=title,
            )

            return "I've scheduled the call for you. You'll receive a call at the specified time."

        except Exception as error:
            print(f"Error scheduling workflow: {error}")
            return "I encountered an error while trying to schedule the call. Please try again later."

    @function_tool
    async def get_scheduled_tasks(
        self,
        context: RunContext,
    ):
        """Get all scheduled tasks for the current user.

        This tool retrieves all workflows associated with the current user from n8n.

        Returns:
            A list of scheduled tasks with their details.
        """
        try:
            # Get the user ID from the context
            participant_identity = next(
                iter(get_job_context().room.remote_participants)
            )

            # Get user's workflows
            workflows = await get_user_workflows(participant_identity)
            
            # Format the response
            tasks = []
            for workflow in workflows:
                # Extract relevant information from the workflow
                task = {
                    "id": workflow["id"],
                    "name": workflow["name"],
                    "active": workflow["active"],
                    "created_at": workflow["createdAt"],
                }
                tasks.append(task)
            
            return tasks

        except Exception as error:
            print(f"Error getting scheduled tasks: {error}")
            return "I encountered an error while trying to get your scheduled tasks. Please try again later."

    @function_tool
    async def delete_scheduled_task(
        self,
        context: RunContext,
        workflow_id: str,
    ):
        """Delete a scheduled task.

        This tool deletes a scheduled workflow from n8n using its workflow ID.
        It first verifies that the workflow belongs to the current user.

        Args:
            workflow_id: The ID of the workflow to delete.

        Returns:
            A string indicating whether the task was successfully deleted.
        """
        try:
            # Get the user ID from the context
            participant_identity = next(
                iter(get_job_context().room.remote_participants)
            )

            # Get user's workflows to verify ownership
            workflows = await get_user_workflows(participant_identity)
            workflow_ids = [w["id"] for w in workflows]
            
            if workflow_id not in workflow_ids:
                return "I couldn't find that scheduled task. Please make sure you're trying to delete one of your own tasks."
            
            await delete_scheduled_workflow(workflow_id)
            return "I've successfully deleted the scheduled task."

        except Exception as error:
            print(f"Error deleting scheduled task: {error}")
            return "I encountered an error while trying to delete the scheduled task. Please try again later."


async def entrypoint(ctx: JobContext):
    # Connect to LiveKit
    await ctx.connect()
    participant = await ctx.wait_for_participant()

    attributes = participant.attributes

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

    if attributes.get("initialRequest"):
        initial_context.add_message(
            role="user",
            content=f"""
                Here's what I want to discuss with you:

                <user_request>
                {attributes["initialRequest"]}
                </user_request>
            """,
        )

    phone_number = attributes.get("sip.phoneNumber")
    
    print("phone_number", phone_number)

    if phone_number is not None:
        initial_context.add_message(
            role="user",
            content=f"""
                Here's the phone number of the user:
                {phone_number}
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
