"""Peewee models for calls and their chronological transcripts."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from peewee import AutoField, ForeignKeyField, Model, SqliteDatabase, TextField


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


db = SqliteDatabase(None, timeout=10, pragmas={'foreign_keys': 1})


class BaseModel(Model):
    class Meta:
        database = db


class Call(BaseModel):
    id = AutoField()
    phone_number = TextField()
    # Keep ISO-formatted UTC strings compatible with existing databases/JSON.
    started_at = TextField(default=now)
    answered_at = TextField(null=True)
    ended_at = TextField(null=True)
    status = TextField(default='ringing')
    end_reason = TextField(null=True)
    error = TextField(null=True)

    class Meta:
        table_name = 'calls'


class Message(BaseModel):
    id = AutoField()
    call = ForeignKeyField(Call, backref='messages', column_name='call_id', index=False)
    timestamp = TextField(default=now)
    role = TextField()
    content = TextField()
    delivery = TextField(default='received')

    class Meta:
        table_name = 'messages'


Message.add_index(Message.call, Message.id, name='messages_call')


def init_db(path: str | Path) -> None:
    """Configure the application database before starting the worker threads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not db.is_closed():
        db.close()
    db.init(str(path))
    with db.connection_context():
        db.create_tables([Call, Message], safe=True)


@contextmanager
def connection():
    # Peewee keeps connections per thread; close them after each operation.
    with db.connection_context():
        with db.atomic():
            yield db


def recover_calls():
    with connection():
        Message.update(delivery='interrupted').where(Message.delivery == 'pending').execute()
        (Call.update(ended_at=now(), status='interrupted',
                     end_reason='previous_process_stopped')
         .where(Call.ended_at.is_null()).execute())


def create_call(phone: str) -> int:
    with connection():
        return Call.create(phone_number=phone).id


def update_caller(call_id: int, phone: str):
    with connection():
        Call.update(phone_number=phone).where(Call.id == call_id).execute()


def answer_call(call_id: int):
    with connection():
        (Call.update(answered_at=now(), status='answered')
         .where(Call.id == call_id).execute())


def finish_call(call_id: int, status: str, reason: str, error: str = ''):
    with connection():
        (Call.update(ended_at=now(), status=status, end_reason=reason, error=error)
         .where((Call.id == call_id) & Call.ended_at.is_null()).execute())


def create_message(call_id: int, role: str, content: str, delivery: str = 'received') -> int:
    with connection():
        return Message.create(call=call_id, role=role, content=content, delivery=delivery).id


def update_delivery(message_id: int, delivery: str):
    with connection():
        (Message.update(delivery=delivery)
         .where(Message.id == message_id).execute())


def get_history(call_id: int, limit: int = 20) -> list[dict]:
    with connection():
        rows = list(Message.select(Message.role, Message.content)
                    .where((Message.call == call_id) &
                           Message.delivery.in_(('received', 'played')))
                    .order_by(Message.id.desc()).limit(limit).dicts())
    return list(reversed(rows))


def get_calls(limit: int = 20):
    with connection():
        return list(Call.select().order_by(Call.id.desc()).limit(limit).dicts())


def get_transcript(call_id: int):
    with connection():
        return list(Message.select(Message.id, Message.call.alias('call_id'),
                                   Message.timestamp, Message.role, Message.content,
                                   Message.delivery)
                    .where(Message.call == call_id).order_by(Message.id).dicts())
