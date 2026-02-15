import asyncio
import json
import os
import time
from functools import partial

import httpx
from livekit.agents import (
    Agent,
    ChatContext,
    ChatMessage,
    RunContext,
    function_tool,
    get_job_context,
)
from zep_cloud.client import Zep

from lib.n8n import (
    create_scheduled_workflow,
    delete_scheduled_workflow,
    get_user_workflows,
)
from prompts import load_system_prompt

zep = Zep(
    api_key=os.getenv("ZEP_API_KEY"),
)

# Shared HTTP client for external API calls (Perplexity, TMDB)
_ext_client: httpx.AsyncClient | None = None


def _get_ext_client() -> httpx.AsyncClient:
    global _ext_client
    if _ext_client is None or _ext_client.is_closed:
        _ext_client = httpx.AsyncClient(
            timeout=20.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _ext_client


async def _run_sync(func, *args, **kwargs):
    """Run a blocking/sync function in a thread executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


class CompanionAgent(Agent):
    session_id: str
    user: dict

    def __init__(self, chat_ctx: ChatContext, session_id: str, user: dict) -> None:
        # Tiny system prompt — processed every turn, so keep it minimal
        system_prompt = load_system_prompt(user_name=user["name"])

        super().__init__(
            chat_ctx=chat_ctx,
            instructions=system_prompt,
        )

        self.session_id = session_id
        self.user = user

    # Ingest messages into memory when the user turns are completed
    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        if not self.session_id:
            return new_message

        messages = turn_ctx.items
        last_message = new_message
        second_to_last_message = messages[-1] if len(messages) >= 2 else None

        if (
            last_message.role == "user"
            and second_to_last_message
            and second_to_last_message.role == "assistant"
        ):
            # Convert messages to the format needed for ingestion
            messages_to_ingest = []
            for message in [last_message, second_to_last_message]:
                role_type = "user" if message.role == "user" else "assistant"
                content = (
                    f"{self.user['name'] if self.user['name'] else 'Unknown Caller'}: {message.text_content}"
                    if role_type == "user"
                    else message.text_content
                )
                messages_to_ingest.append(
                    {
                        "content": content,
                        "role_type": role_type,
                    }
                )

            # Run memory ingestion in background without waiting
            asyncio.create_task(self._ingest_messages_background(messages_to_ingest))

        return new_message

    async def _ingest_messages_background(self, messages_to_ingest: list) -> None:
        """Background task to ingest messages into memory (runs sync Zep call in thread)."""
        try:
            start_time = time.monotonic()
            await _run_sync(
                zep.memory.add,
                self.session_id,
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
    async def movie_recommendation(
        self,
        context: RunContext,
        query: str,
        genre: str = "",
        media_type: str = "both",
    ):
        """Search for movies and TV shows available on streaming platforms in the Netherlands.

        Use this tool when the user asks for something to watch, mentions movies or series
        they like, wants entertainment recommendations, or during calm evening conversations.

        Args:
            query: What to search for. Can be a title, topic, or description like "Dutch thriller" or "nature documentary".
            genre: Optional genre filter like "drama", "comedy", "documentary", "thriller", "romance", "animation".
            media_type: Type of content: "movie", "tv", or "both".

        Returns:
            A string with movie/show recommendations including streaming availability.
        """
        await context.session.generate_reply(
            instructions=f'Tell the user briefly (one short sentence in Dutch) that you\'re checking what\'s available for "{query}".'
        )

        tmdb_api_key = os.getenv("TMDB_API_KEY")
        if not tmdb_api_key:
            print("[MovieRec] No TMDB_API_KEY, falling back to web search")
            return await self._movie_search_fallback(query, genre)

        try:
            start_time = time.monotonic()
            results = []
            client = _get_ext_client()

            # Search TMDB
            search_type = "multi" if media_type == "both" else media_type
            search_url = f"https://api.themoviedb.org/3/search/{search_type}"

            search_response = await client.get(
                search_url,
                params={
                    "api_key": tmdb_api_key,
                    "query": query,
                    "language": "nl-NL",
                    "region": "NL",
                    "include_adult": "false",
                },
            )

            if search_response.status_code != 200:
                print(f"[MovieRec] TMDB search failed: {search_response.status_code}")
                return await self._movie_search_fallback(query, genre)

            search_data = search_response.json()
            items = search_data.get("results", [])[:5]  # Reduced from 8 to 5 for speed

            # Fetch streaming providers in parallel
            async def get_item_with_providers(item):
                item_type = item.get("media_type", media_type if media_type != "both" else "movie")
                if item_type not in ("movie", "tv"):
                    return None

                item_id = item["id"]
                title = item.get("title") or item.get("name", "Unknown")
                overview = item.get("overview", "")[:200]
                rating = item.get("vote_average", 0)
                year = (item.get("release_date") or item.get("first_air_date") or "")[:4]

                streaming_platforms = []
                try:
                    providers_url = f"https://api.themoviedb.org/3/{item_type}/{item_id}/watch/providers"
                    prov_response = await client.get(
                        providers_url,
                        params={"api_key": tmdb_api_key},
                    )
                    if prov_response.status_code == 200:
                        nl_data = prov_response.json().get("results", {}).get("NL", {})
                        for provider in nl_data.get("flatrate", []):
                            streaming_platforms.append(provider["provider_name"])
                        for provider in nl_data.get("free", []):
                            streaming_platforms.append(f"{provider['provider_name']} (gratis)")
                except Exception:
                    pass

                return {
                    "title": title,
                    "year": year,
                    "type": "Film" if item_type == "movie" else "Serie",
                    "rating": f"{rating:.1f}/10" if rating > 0 else "Geen score",
                    "description": overview,
                    "streaming": streaming_platforms if streaming_platforms else ["Niet gevonden op streaming"],
                }

            # Parallel provider lookups
            tasks = [get_item_with_providers(item) for item in items]
            results = [r for r in await asyncio.gather(*tasks) if r is not None]

            end_time = time.monotonic()
            print(f"[MovieRec] TMDB search took: {end_time - start_time:.2f} seconds, found {len(results)} results")

            if not results:
                return f"No results found for '{query}'. Try a different search term."

            output = f"Entertainment recommendations for '{query}' (Netherlands):\n\n"
            for i, r in enumerate(results[:5], 1):
                platforms = ", ".join(r["streaming"])
                output += f"{i}. {r['title']} ({r['year']}) - {r['type']}\n"
                output += f"   Score: {r['rating']}\n"
                output += f"   Beschikbaar op: {platforms}\n"
                if r["description"]:
                    output += f"   {r['description']}\n"
                output += "\n"

            return output

        except Exception as error:
            print(f"[MovieRec] Error: {error}")
            return await self._movie_search_fallback(query, genre)

    async def _movie_search_fallback(self, query: str, genre: str = "") -> str:
        """Fallback: use Perplexity web search for movie recommendations (async)."""
        search_query = f"beste {genre} films series op Netflix Amazon Prime NPO Nederland 2026: {query}"
        try:
            client = _get_ext_client()
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                json={
                    "messages": [{"content": search_query, "role": "user"}],
                    "model": "sonar",
                },
                headers={
                    "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
                    "Content-Type": "application/json",
                },
            )
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return f"Entertainment search results (web):\n{content}"
        except Exception as e:
            print(f"[MovieRec] Fallback also failed: {e}")

        return "I couldn't find entertainment recommendations right now. Try asking me later!"

    @function_tool
    async def web_search(
        self,
        context: RunContext,
        query: str,
    ):
        """Search the web for information.

        Use this tool when the user asks for information that requires up-to-date knowledge.

        Args:
            query: The search query to look up information for. Be specific and concise.

        Returns:
            A string containing the search results and relevant information.
        """

        await context.session.generate_reply(
            instructions=f'Tell the user very briefly (one short sentence in Dutch) that you\'re looking up "{query}".'
        )

        try:
            start_time = time.monotonic()
            client = _get_ext_client()
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                json={
                    "messages": [{"content": query, "role": "user"}],
                    "model": "sonar",
                },
                headers={
                    "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
                    "Content-Type": "application/json",
                },
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

            json_data = json.dumps(data)

            if participant_identity.startswith("sip_"):
                return json_data
            else:
                start_time = time.monotonic()
                result = await get_job_context().room.local_participant.perform_rpc(
                    destination_identity=participant_identity,
                    method="web_search",
                    payload=json_data,
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
        """Schedule a reminder notification as a push notification. Use schedule_task for phone call reminders.

        Always use get_local_time first to get the user's current time.

        Args:
            repeats: Whether the notification should repeat.
            weekDay: Day of the week (1=Sunday, 7=Saturday).
            day: Day of the month.
            year: Year.
            hour: Hour (0-23).
            minute: Minute (0-59).
            month: Month (1-12).
            message: The reminder message.
            title: The notification title.
        """
        try:
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
            print(f"RPC schedule_reminder_notification took: {end_time - start_time:.2f} seconds")
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
        """Schedule a phone call at a specific time to discuss a topic.

        Args:
            cron_expression: Cron expression for when to trigger.
            title: Title of the task.
            message: Topic to discuss during the call.

        Returns:
            Confirmation string.
        """
        try:
            participant_identity = next(
                iter(get_job_context().room.remote_participants)
            )

            start_time = time.monotonic()
            await create_scheduled_workflow(
                cron=cron_expression,
                phone_number=self.user["phoneNumber"],
                user_id=participant_identity,
                message=message,
                title=title,
            )
            end_time = time.monotonic()
            print(f"N8n create_scheduled_workflow took: {end_time - start_time:.2f} seconds")

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

        Returns:
            A list of scheduled tasks with their details.
        """
        try:
            participant_identity = next(
                iter(get_job_context().room.remote_participants)
            )

            start_time = time.monotonic()
            workflows = await get_user_workflows(participant_identity)
            end_time = time.monotonic()
            print(f"N8n get_user_workflows took: {end_time - start_time:.2f} seconds")

            tasks = []
            for workflow in workflows:
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
        """Delete a scheduled task by its workflow ID.

        Args:
            workflow_id: The ID of the workflow to delete.

        Returns:
            Confirmation string.
        """
        try:
            participant_identity = next(
                iter(get_job_context().room.remote_participants)
            )

            start_time = time.monotonic()
            workflows = await get_user_workflows(participant_identity)
            end_time = time.monotonic()
            print(f"N8n get_user_workflows (for deletion check) took: {end_time - start_time:.2f} seconds")
            workflow_ids = [w["id"] for w in workflows]

            if workflow_id not in workflow_ids:
                return "I couldn't find that scheduled task. Please make sure you're trying to delete one of your own tasks."

            start_time = time.monotonic()
            await delete_scheduled_workflow(workflow_id)
            end_time = time.monotonic()
            print(f"N8n delete_scheduled_workflow took: {end_time - start_time:.2f} seconds")
            return "I've successfully deleted the scheduled task."

        except Exception as error:
            print(f"Error deleting scheduled task: {error}")
            return "I encountered an error while trying to delete the scheduled task. Please try again later."
