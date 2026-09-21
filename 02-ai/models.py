from __future__ import annotations

from datetime import datetime

from peewee import (
    AutoField,
    CharField,
    DateTimeField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
)

# Read database path from config with fallback default
try:
    from config import DB_PATH  # type: ignore
except (ImportError, AttributeError):
    DB_PATH = 'conversations.db'

# Configure SQLite database with WAL mode and foreign key support
db = SqliteDatabase(
    DB_PATH,
    pragmas={
        'journal_mode': 'wal',
        'foreign_keys': 1,
        'ignore_check_constraints': 0,
    },
)


class BaseModel(Model):
    class Meta:
        database = db


class Conversation(BaseModel):
    """Conversation session model."""

    id = AutoField()
    title = CharField(max_length=255, default='New Conversation')
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = 'conversations'
        order_by = ('-updated_at',)

    def add_message(self, role: str, content: str, tokens: int | None = None) -> Message:
        """Append a message to this conversation and update the timestamp."""
        message = Message.create(conversation=self, role=role, content=content, tokens=tokens)
        self.updated_at = datetime.now()
        self.save(only=[Conversation.updated_at])
        return message

    def get_messages(self, limit: int | None = None) -> list[Message]:
        """Retrieve conversation messages ordered chronologically."""
        query = self.messages.order_by(Message.created_at.asc())
        if limit is not None:
            total = self.messages.count()
            if total > limit:
                query = query.offset(total - limit)
        return list(query)

    def to_openai_format(self, system_prompt: str | None = None, limit: int | None = None) -> list[dict[str, str]]:
        """Convert conversation history into standard OpenAI messages format.

        Format: [{'role': 'user'|'assistant'|'system', 'content': '...'}]
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        for msg in self.get_messages(limit=limit):
            if msg.role == 'system' and system_prompt:
                continue
            messages.append({'role': msg.role, 'content': msg.content})

        return messages


class Message(BaseModel):
    """Message model for storing individual chat messages."""

    id = AutoField()
    conversation = ForeignKeyField(Conversation, backref='messages', on_delete='CASCADE')
    role = CharField(max_length=20)  # system, user, assistant
    content = TextField()
    tokens = IntegerField(null=True)
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = 'messages'
        order_by = ('created_at',)


def init_db(db_path: str | None = None) -> None:
    """Initialize database connection and ensure tables exist."""
    if db_path is not None:
        db.init(db_path, pragmas={'journal_mode': 'wal', 'foreign_keys': 1})
    db.connect(reuse_if_open=True)
    db.create_tables([Conversation, Message], safe=True)


def create_conversation(title: str = 'New Conversation') -> Conversation:
    """Create and persist a new conversation session."""
    init_db()
    return Conversation.create(title=title)


def get_conversation(conversation_id: int) -> Conversation | None:
    """Retrieve a conversation by its primary key ID."""
    init_db()
    return Conversation.get_or_none(Conversation.id == conversation_id)


def get_all_conversations(limit: int = 50) -> list[Conversation]:
    """Retrieve recent conversations ordered by most recently updated."""
    init_db()
    return list(Conversation.select().order_by(Conversation.updated_at.desc()).limit(limit))


def delete_conversation(conversation_id: int) -> bool:
    """Delete a conversation and cascade-delete its messages."""
    conv = get_conversation(conversation_id)
    if conv:
        conv.delete_instance(recursive=True)
        return True
    return False


# Auto-initialize database on module import
init_db()

if __name__ == '__main__':
    print('Testing models and SQLite database...')
    init_db()

    conv = create_conversation(title='Test Session')
    print(f'Created Conversation ID: {conv.id}')

    conv.add_message(role='user', content='Hello, how are you?')
    conv.add_message(role='assistant', content="I'm doing well, thank you!")

    chat_history = conv.to_openai_format(system_prompt='You are a helpful assistant.')
    print(chat_history)
    print('\nFormatted messages for OpenAI/LLM:')
    for item in chat_history:
        print(f"  [{item['role']}]: {item['content']}")
