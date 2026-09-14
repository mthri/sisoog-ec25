"""Database models for call logs and SMS management."""

from datetime import datetime
from peewee import CharField, DateTimeField, Model, SqliteDatabase, TextField

db = SqliteDatabase('modem.db')


class BaseModel(Model):
    class Meta:
        database = db


class CallLog(BaseModel):
    phone_number = CharField(max_length=32)
    timestamp = DateTimeField(default=datetime.now)
    call_type = CharField(default='incoming')
    status = CharField(default='ringing')


class SMSMessage(BaseModel):
    phone_number = CharField(max_length=32)
    message = TextField()
    direction = CharField(default='inbound')
    timestamp = DateTimeField(default=datetime.now)
    status = CharField(default='received')


def init_db() -> None:
    """Create database tables."""
    with db:
        db.create_tables([CallLog, SMSMessage])