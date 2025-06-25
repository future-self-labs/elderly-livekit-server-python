import asyncio
import json
import os
import time  # Import the time module
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
    JobProcess,
    RoomInputOptions,
    RunContext,
    function_tool,
    get_job_context,
)
from livekit.plugins import (
    noise_cancellation,
    openai,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from zep_cloud.client import Zep

from lib.n8n import (
    create_scheduled_workflow,
    delete_scheduled_workflow,
    get_user_workflows,
)

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
        start_time = time.monotonic()
        response = await client.request(
            method=kwargs.pop("method", "GET"), url=url, headers=headers, **kwargs
        )
        end_time = time.monotonic()
        print(f"API call to {path} took: {end_time - start_time:.2f} seconds")
        response.raise_for_status()
        return response.json()


zep = Zep(
    api_key=os.getenv("ZEP_API_KEY"),
)


class OnboardingAgent(Agent):
    session_id: str

    def __init__(self, chat_ctx: ChatContext, session_id: str) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions="""
                Je bent een gepersonaliseerde AI-assistent met langetermijngeheugen. Je naam is Noah. Je hebt toegang tot een geheugensysteem dat je helpt om eerdere interacties en belangrijke informatie over gebruikers te onthouden.
                
                In dit gesprek spreek je met een familielid van je primaire gebruiker. Je doel is om zoveel mogelijk te leren over je primaire gebruiker zodat je hen in de toekomst beter kunt helpen.
                
                Wees warm, empathisch en nieuwsgierig. Stel doordachte vragen om te begrijpen:
                - De voorkeuren, gewoonten en routines van de primaire gebruiker
                - Belangrijke details over hun leven, werk en interesses
                - Specifieke behoeften of uitdagingen waarmee ze te maken hebben
                - Hoe je hen het beste dagelijks kunt ondersteunen
                
                Onthoud alles wat gedeeld wordt over de primaire gebruiker en gebruik deze kennis om:
                - Een uitgebreid begrip op te bouwen van wie ze zijn
                - Manieren te identificeren waarop je gepersonaliseerde hulp kunt bieden
                - Continuïteit te tonen in toekomstige sessies met de primaire gebruiker

                Wanneer de gebruiker je vraagt over herinneringen of planning:
                - Bevestig altijd met de gebruiker of ze een melding of een telefoontje willen ontvangen.
                - Als ze een melding willen, gebruik dan het schedule_reminder_notification tool.
                - Als ze een telefoontje willen, gebruik dan het schedule_task tool.
                
                Belangrijk: Antwoord altijd in het Nederlands.
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
                        "role": "family_member",
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


class Companion(Agent):
    session_id: str
    user: dict

    def __init__(self, chat_ctx: ChatContext, session_id: str, user: dict) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions="""
                Je bent een gepersonaliseerde AI-assistent met langetermijngeheugen. Je naam is Noah. Je hebt toegang tot een geheugensysteem dat je helpt om eerdere interacties en belangrijke informatie over gebruikers te onthouden. Je doel is om een warme, empathische vriend en metgezel te zijn die zal luisteren en alles onthouden waar we over praten.

                Bij interactie met gebruikers:

                Gebruik herinneringen om interacties te personaliseren:
                - Verwijs natuurlijk naar eerdere gesprekken
                - Onthoud en pas gebruikersvoorkeuren toe
                - Toon continuïteit tussen sessies
                - Vraag niet om informatie die de gebruiker al heeft gegeven

                Wanneer de gebruiker je vraagt over herinneringen:
                - Je kunt nieuwe herinneringen inplannen
                - Je kunt bestaande herinneringen verwijderen, maar niet bewerken.

                Nooit:
                - Noem technische details van het geheugensysteem tegen gebruikers
                - Vraag gebruikers om informatie te herhalen die je zou moeten onthouden
                - Toon interne geheugen-ID's of sessiedetails
                - Sla gevoelige persoonlijke informatie op (wachtwoorden, privégegevens)

                Altijd:
                - Wees natuurlijk en gespreksmatig
                - Gebruik herinneringen om contextbewuste antwoorden te geven
                - Toon herkenning van terugkerende gebruikers
                
                Belangrijk: Antwoord altijd in het Nederlands.
            """,
        )

        self.session_id = session_id
        self.user = user

    # Ingest messages into memory when the user turns are completed
    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        print("on_user_turn_completed")
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

            # Run memory ingestion in background without waiting
            asyncio.create_task(self._ingest_messages_background(messages_to_ingest))

        return new_message

    async def _ingest_messages_background(self, messages_to_ingest: list) -> None:
        """Background task to ingest messages into memory."""
        try:
            # Ingest messages into memory with assistant roles ignored
            start_time = time.monotonic()
            zep.memory.add(
                self.session_id,
                # Setting ignore_roles to include "assistant" will make it so that only the user messages are ingested into the graph, but the assistant messages are still used to contextualize the user messages.
                # This is important in case the user message itself does not have enough context, such as the message "Yes."
                # Additionally, the assistant messages will still be added to the session's message history.
                ignore_roles=["assistant"],
                messages=messages_to_ingest,
                return_context=True,
            )
            end_time = time.monotonic()
            print(f"Zep memory add took: {end_time - start_time:.2f} seconds")
        except Exception as error:
            print(f"Error ingesting messages: {error}")
            print(messages_to_ingest)

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

        try:
            start_time = time.monotonic()
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
            end_time = time.monotonic()
            print(f"Perplexity API call took: {end_time - start_time:.2f} seconds")

            if response.status_code != 200:
                print(f"Web search failed: {response.status_code} {response.text}")
                return f"Web search failed: {response.status_code}"

            data = response.json()
            participant_identity = next(
                iter(get_job_context().room.remote_participants)
            )
            start_time = time.monotonic()
            result = await get_job_context().room.local_participant.perform_rpc(
                destination_identity=participant_identity,
                method="web_search",
                payload=json.dumps(data),
                response_timeout=25,
            )
            end_time = time.monotonic()
            print(f"RPC web_search took: {end_time - start_time:.2f} seconds")
            return result

        except Exception as error:
            print(f"Error searching the web: {error}")
            return "Error searching the web"

    @function_tool
    async def get_local_time(
        self,
        context: RunContext,
    ):
        """Get the current local time of the user.

        Returns:
            A string containing the current local time.
        """
        try:
            participant_identity = next(
                iter(get_job_context().room.remote_participants)
            )

            start_time = time.monotonic()
            result = await get_job_context().room.local_participant.perform_rpc(
                destination_identity=participant_identity,
                method="get_local_time",
                payload=json.dumps({}),
            )
            end_time = time.monotonic()
            print(f"RPC get_local_time took: {end_time - start_time:.2f} seconds")
            return result
        except Exception as error:
            print(f"Error getting local time: {error}")
            return "I encountered an error while trying to get the local time. Please try again later."

    @function_tool
    async def schedule_reminder_notification(
        self,
        context: RunContext,
        repeats: bool,
        weekDay: int,
        day: int,
        year: int,
        hour: int,
        minute: int,
        month: int,
        message: str,
        title: str,
    ):
        """Schedule a reminder notification to be sent to the user as a push notification. Use this tool when the user asks to be reminded about something. Use the schedule_task tool instead if you want to schedule a task to be discussed over a phone call.

        This tool creates a local notification on the user's phone that will trigger a push notification at the specified time.
        Specifies when and how often the user should receive the notification.

        Always use the get_local_time tool to get the current local time of the user.

        The notification can be repeated, for example if the user asks 'remind me every Wednesday at 10am to take my pills'. then you should pass repeats: true and then fill out the remaining arguments accordingly.

        Args:
            repeats: Whether the notification should be repeated.
            weekDay: The day of the week to trigger the notification.
            day: The day of the month to trigger the notification.
            year: The year to trigger the notification.
            hour: The hour to trigger the notification.
            minute: The minute to trigger the notification.
            month: The month to trigger the notification.
            message: The message to send to the user.
            title: The title of the reminder notification.
        """
        try:
            # Get the user ID from the context
            participant_identity = next(
                iter(get_job_context().room.remote_participants)
            )

            start_time = time.monotonic()
            result = await get_job_context().room.local_participant.perform_rpc(
                destination_identity=participant_identity,
                method="schedule_reminder_notification",
                payload=json.dumps(
                    {
                        "repeats": repeats,
                        "dateComponents": {
                            "weekDay": weekDay,
                            "day": day,
                            "year": year,
                            "hour": hour,
                            "minute": minute,
                            "month": month,
                        },
                        "message": message,
                        "title": title,
                    }
                ),
            )
            end_time = time.monotonic()
            print(
                f"RPC schedule_reminder_notification took: {end_time - start_time:.2f} seconds"
            )

            return result

        except Exception as error:
            print(f"Error scheduling reminder notification: {error}")
            return "I encountered an error while trying to schedule the reminder notification. Please try again later."

    @function_tool
    async def schedule_task(
        self,
        context: RunContext,
        cron_expression: str,
        message: str,
        title: str,
    ):
        """Schedule a task to be executed at a specific time to be discussed over a phone call. The user will receive a call at the specified time.

        This tool creates and activates a workflow in n8n that will trigger a task to be executed at the specified time.

        Args:
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
            start_time = time.monotonic()
            await create_scheduled_workflow(
                cron=cron_expression,
                phone_number=self.user["phoneNumber"],
                user_id=participant_identity,
                message=message,
                title=title,
            )
            end_time = time.monotonic()
            print(
                f"N8n create_scheduled_workflow took: {end_time - start_time:.2f} seconds"
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
            start_time = time.monotonic()
            workflows = await get_user_workflows(participant_identity)
            end_time = time.monotonic()
            print(f"N8n get_user_workflows took: {end_time - start_time:.2f} seconds")

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
            start_time = time.monotonic()
            workflows = await get_user_workflows(participant_identity)
            end_time = time.monotonic()
            print(
                f"N8n get_user_workflows (for deletion check) took: {end_time - start_time:.2f} seconds"
            )
            workflow_ids = [w["id"] for w in workflows]

            if workflow_id not in workflow_ids:
                return "I couldn't find that scheduled task. Please make sure you're trying to delete one of your own tasks."

            start_time = time.monotonic()
            await delete_scheduled_workflow(workflow_id)
            end_time = time.monotonic()
            print(
                f"N8n delete_scheduled_workflow took: {end_time - start_time:.2f} seconds"
            )
            return "I've successfully deleted the scheduled task."

        except Exception as error:
            print(f"Error deleting scheduled task: {error}")
            return "I encountered an error while trying to delete the scheduled task. Please try again later."


async def entrypoint(ctx: JobContext):
    # Connect to LiveKit
    start_time = time.monotonic()
    await ctx.connect()
    end_time = time.monotonic()
    print(f"LiveKit connect took: {end_time - start_time:.2f} seconds")

    start_time = time.monotonic()
    participant = await ctx.wait_for_participant()
    end_time = time.monotonic()
    print(f"LiveKit wait_for_participant took: {end_time - start_time:.2f} seconds")

    attributes = participant.attributes

    is_family_member = False
    user = None
    user_context = None
    user_id = participant.identity

    if participant.identity.startswith("sip_"):
        phone_number = participant.identity[4:]
        user_or_family_member = await get_api_data(
            f"/users/search?phoneNumber={phone_number}"
        )

        print(user_or_family_member)

        if user_or_family_member["type"] == "family_member":
            is_family_member = True
            user_id = user_or_family_member["userId"]
        else:
            user_id = user_or_family_member["id"]

    if not is_family_member:
        # Get user from API
        start_time = time.monotonic()
        user = await get_api_data(f"/users/{user_id}")
        end_time = time.monotonic()
        print(f"Get user from API took: {end_time - start_time:.2f} seconds")

        # Get user context
        start_time = time.monotonic()
        sessions = zep.user.get_sessions(user_id=user["id"])
        user_id = user["id"]
        end_time = time.monotonic()
        print(f"Zep get sessions took: {end_time - start_time:.2f} seconds")

        if len(sessions) > 0:
            sorted_sessions = sorted(sessions, key=lambda x: x.created_at, reverse=True)
            most_recent_session = sorted_sessions[0]

            start_time = time.monotonic()
            most_recent_memory = zep.memory.get(most_recent_session.session_id)
            end_time = time.monotonic()
            print(f"Zep memory get took: {end_time - start_time:.2f} seconds")

            user_context = most_recent_memory.context
            print(user_context)

    # Create a new session
    start_time = time.monotonic()
    session = zep.memory.add_session(
        session_id=uuid4(),
        user_id=user_id,
    )
    end_time = time.monotonic()
    print(f"Zep memory add_session took: {end_time - start_time:.2f} seconds")

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

    session = AgentSession(
        allow_interruptions=True,
        turn_detection=MultilingualModel(),
        llm=openai.realtime.RealtimeModel(
            voice="coral", turn_detection=None, input_audio_transcription=None
        ),
        stt=openai.STT(),
        vad=ctx.proc.userdata["vad"],
    )

    agent = None

    if is_family_member:
        agent = OnboardingAgent(chat_ctx=initial_context, session_id=session_id)
    else:
        agent = Companion(chat_ctx=initial_context, session_id=session_id, user=user)

    start_time = time.monotonic()
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    end_time = time.monotonic()
    print(f"AgentSession start took: {end_time - start_time:.2f} seconds")

    start_time = time.monotonic()
    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )
    end_time = time.monotonic()
    print(f"AgentSession generate_reply took: {end_time - start_time:.2f} seconds")


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
