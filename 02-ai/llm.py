from __future__ import annotations

from collections.abc import Generator

from openai import OpenAI

from models import (
    Conversation,
    Message,
    create_conversation,
    delete_conversation,
    get_all_conversations,
    get_conversation,
)


BASE_URL = 'http://localhost:11434/v1'
MODEL_NAME = 'gemma4:e2b'
API_KEY = 'ollama'
SYSTEM_PROMPT = """
تو یک دستیار صوتی فارسی‌زبان هستی. مثل یک دوست خوش‌برخورد، طبیعی و خودمونی حرف بزن.
پاسخ پیش‌فرضت فقط یک یا دو جملهٔ کوتاه باشه؛ فقط وقتی کاربر توضیح بیشتر خواست، مفصل‌تر جواب بده.
مستقیم برو سر اصل مطلب. سؤال کاربر رو تکرار نکن و مقدمه، تعارف و جمع‌بندی اضافه نگو.
از فارسی محاوره‌ای مثل «می‌تونی»، «اگه» و «باشه» استفاده کن؛ لحن اداری و کتابی نداشته باش.
جوابت قراره با صدا پخش بشه: متن ساده با جمله‌های کوتاه بنویس، بدون مارک‌داون، فهرست، ایموجی و نشانه‌های تزئینی.
تو بخش گفتگویی یک دستیار صوتی هستی؛ متن خروجی تو به‌صورت خودکار با موتور تبدیل متن به گفتار برای کاربر خوانده می‌شه.
وقتی کاربر می‌گه «شعر بخون»، «قصه بگو» یا «این متن رو بخون»، خود متن درخواستی رو بنویس تا پخش بشه؛ نگو «من فقط مدل متنی هستم» یا «نمی‌تونم صدا تولید کنم».
برای شعر، چند بیت کوتاه بنویس و هر مصراع رو در یک خط جدا بذار. اگه شعر رو خودت می‌سازی، به شاعر واقعی نسبتش نده.
خواندن متن با TTS به معنی آواز یا اجرای موسیقی نیست؛ ادعای اجرای آهنگ و ملودی نکن.
هر بار سلام نکن، خودت رو معرفی نکن و آخر هر جواب پیشنهاد کمک بیشتر نده.
اگه درخواست مبهم بود، فقط یک سؤال کوتاه برای روشن شدنش بپرس. اگه چیزی رو نمی‌دونی، کوتاه و صادقانه بگو.
""".strip()


client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


class ChatManager:
    """Manages chat interactions with an LLM and persists conversation history in SQLite."""

    def __init__(
        self,
        conversation: Conversation | int | None = None,
        title: str = 'New Conversation',
        system_prompt: str = SYSTEM_PROMPT,
        model: str = MODEL_NAME,
        client: OpenAI = client,
        max_history: int | None = None,
        enable_thinking: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.enable_thinking = enable_thinking

        if isinstance(conversation, Conversation):
            self.conversation = conversation
        elif isinstance(conversation, int):
            found = get_conversation(conversation)
            if found is None:
                raise ValueError(f'Conversation with ID {conversation} not found')
            self.conversation = found
        else:
            self.conversation = create_conversation(title=title)

    @property
    def conversation_id(self) -> int:
        """Return the current conversation ID."""
        return self.conversation.id

    @property
    def title(self) -> str:
        """Return the conversation title."""
        return self.conversation.title

    def set_title(self, title: str) -> None:
        """Update conversation title in the database."""
        self.conversation.title = title
        self.conversation.save(only=[Conversation.title])

    def get_messages(self) -> list[Message]:
        """Return message objects for current conversation."""
        return self.conversation.get_messages(limit=self.max_history)

    def get_history(self, include_system: bool = True) -> list[dict[str, str]]:
        """Return conversation messages formatted for OpenAI chat completion."""
        sys_prompt = self.system_prompt if include_system else None
        return self.conversation.to_openai_format(system_prompt=sys_prompt, limit=self.max_history)

    def _request(self, user_message: str, enable_thinking: bool | None, **kwargs):
        """Save the user message and send the same options for both response modes."""
        if self.conversation.title == 'New Conversation':
            clean_title = user_message.strip()[:40]
            if clean_title:
                self.set_title(clean_title)

        self.conversation.add_message(role='user', content=user_message)
        messages = self.get_history(include_system=True)

        effective_thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        extra_body = kwargs.pop('extra_body', {})
        extra_body.setdefault('think', effective_thinking)
        extra_body.setdefault('enable_thinking', effective_thinking)

        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body=extra_body,
            **kwargs,
        )

    def ask(self, user_message: str, enable_thinking: bool | None = None, **kwargs) -> str:
        """Request a complete reply and save it to the conversation."""
        response = self._request(user_message, enable_thinking, **kwargs)

        reply = response.choices[0].message.content or ''
        tokens = response.usage.total_tokens if response.usage else None
        self.conversation.add_message(role='assistant', content=reply, tokens=tokens)
        return reply

    def ask_stream(self, user_message: str, enable_thinking: bool | None = None, **kwargs) -> Generator[str, None, str]:
        """Stream LLM response chunks, accumulate full response, and persist to database."""
        stream_response = self._request(
            user_message, enable_thinking, stream=True, **kwargs,
        )

        in_thinking = False
        full_content: list[str] = []
        for chunk in stream_response:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                full_content.append(delta)
                if '<think>' in delta:
                    in_thinking = True
                if '</think>' in delta:
                    in_thinking = False
                    after_tag = delta.split('</think>', 1)[1]
                    if after_tag:
                        yield after_tag
                    continue
                if not in_thinking:
                    yield delta

        reply = ''.join(full_content)
        self.conversation.add_message(role='assistant', content=reply)
        return reply

    def reset(self, title: str = 'New Conversation') -> Conversation:
        """Start a new conversation session."""
        self.conversation = create_conversation(title=title)
        return self.conversation

    @classmethod
    def load(cls, conversation_id: int, **kwargs) -> ChatManager:
        """Load an existing conversation by ID."""
        return cls(conversation=conversation_id, **kwargs)

    @staticmethod
    def list_conversations(limit: int = 50) -> list[Conversation]:
        """List recent conversations."""
        return get_all_conversations(limit=limit)

    @staticmethod
    def delete_conversation(conversation_id: int) -> bool:
        """Delete a conversation by ID."""
        return delete_conversation(conversation_id)


ChatSession = ChatManager


def warmup_llm(prompt: str = 'سلام', max_tokens: int = 1, *, strict: bool = False) -> None:
    """Send a lightweight prompt to preload LLM weights into memory."""
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
        )
    except Exception as exc:
        if strict:
            raise
        print(f'[-] Warning: Failed to warm up LLM ({MODEL_NAME}): {exc}')


if __name__ == '__main__':
    print('Testing ChatManager...')
    chat = ChatManager()
    print(f'Started conversation with ID: {chat.conversation_id}')

    print('\nSending first message...')
    response1 = chat.ask('سلام، حالت چطوره؟')
    print(f'Assistant: {response1}')

    print('\nSending follow-up message (streaming)...')
    print('Assistant: ', end='', flush=True)
    for chunk in chat.ask_stream('امروز هوا چطوره؟'):
        print(chunk, end='', flush=True)
    print()

    print(f'\nConversation Title: {chat.title}')
    print('Message history in DB:')
    for msg in chat.get_messages():
        print(f'  [{msg.role}]: {msg.content}')
