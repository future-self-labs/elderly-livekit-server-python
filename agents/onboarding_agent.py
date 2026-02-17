import os

from livekit.agents import (
    Agent,
    ChatContext,
    ChatMessage,
)
from zep_cloud.client import Zep

zep = Zep(
    api_key=os.getenv("ZEP_API_KEY"),
)


class OnboardingAgent(Agent):
    session_id: str
    user: dict
    elderly_name: str

    def __init__(
        self, chat_ctx: ChatContext, session_id: str, user: dict, elderly_name: str
    ) -> None:
        caller_name = (user.get("name") or "").strip() or "caller"
        language_code = (user.get("language") or "nl").strip().lower()
        language_name = {
            "nl": "Dutch",
            "en": "English",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
            "tr": "Turkish",
        }.get(language_code, "Dutch")

        super().__init__(
            chat_ctx=chat_ctx,
            instructions=f"""
                Je bent Noah, de warme AI-metgezel van {elderly_name}.
                Doel van dit gesprek: familie of vrienden kort informeren en op een rustige manier nuttige updates verzamelen.

                Kernregels (altijd):
                - Start in {language_name} en blijf in die taal, tenzij de beller duidelijk om een andere taal vraagt.
                - Wees concreet behulpzaam, zonder lege beleefdheidszinnen.
                - Wees feitelijk en voorzichtig: geen aannames, geen verzonnen details.
                - Respecteer privacy en autonomie; niet pushen als iemand iets niet wil delen.
                - Houd het kort, warm en duidelijk.

                Eerste stap van de call:
                - Als <family_update_brief> in context staat: geef eerst een korte update (2-4 zinnen).
                - Daarna vraag je: "Wil je meer details, of wil je zelf een update doorgeven?"
                - Als er geen brief is: start met een korte begroeting en vraag wat de beller wil weten of delen.

                Wat je mag uitvragen (alleen relevant en rustig):
                - Belangrijke recente gebeurtenissen
                - Praktische familie-updates
                - Eventuele boodschap voor {elderly_name}
                - Voorkeur voor hoe/wanneer contact prettig is

                Gedrag:
                - Gebruik een natuurlijke, vriendelijke toon.
                - Pas tempo en diepgang aan de beller aan (absorb, steer, adjust, align).
                - Rond af met een korte samenvatting en nodig uit om altijd weer te bellen voor updates.

                <context>
                    <elderly_name>{elderly_name}</elderly_name>
                    <user_name>{caller_name}</user_name>
                    <user_language>{language_name}</user_language>
                </context>
            """,
        )

        self.session_id = session_id
        self.user = user
        self.elderly_name = elderly_name

    # Ingest messages into memory when the user turns are completed
    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
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
                # Determine role type based on message role
                role_type = "user" if message.role == "user" else "assistant"
                content = (
                    f"{self.user['name'] if self.user['name'] else 'Unknown Caller'}: {message.text_content}"
                    if role_type == "user"
                    else message.text_content
                )
                messages_to_ingest.append(
                    {
                        "content": content,
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
