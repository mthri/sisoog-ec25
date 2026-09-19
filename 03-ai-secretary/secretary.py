"""The main thread owns call state; one worker owns all AI and audio operations."""
import logging
import queue
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from contextlib import nullcontext

import models

log = logging.getLogger(__name__)


def is_goodbye(text):
    """Recognize standalone farewells, not mentions or negations of goodbye."""
    text = unicodedata.normalize('NFKC', text).translate(str.maketrans('يك', 'یک'))
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^\w\s]|_', ' ', text)
    text = ' '.join(text.split())
    farewell = r'(?:خدا\s*حافظ|خدا\s*نگهدار|خدا\s*نگه\s*دار|بای بای|فعلا)'
    courtesy = (r'(?:ممنون(?:م)?|مرسی|متشکرم|خیلی ممنون(?:م)?|دست شما درد نکنه|'
                r'باشه|خب|خوب|بله|اوکی|دیگه کاری ندارم|کاری ندارم|'
                r'روز(?:تون|تان)? بخیر|شب(?:تون|تان)? بخیر|وقت(?:تون|تان)? بخیر|'
                r'خسته نباشید|به سلامت|موفق باشید)')
    return re.fullmatch(rf'(?:{courtesy} )*{farewell}(?: (?:{courtesy}|{farewell}))*', text) is not None


@dataclass
class Call:
    id: int
    first_ring: float
    last_ring: float
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    worker: threading.Thread | None = None
    answered_at: float | None = None
    ended: bool = False
    reason: str = 'completed'
    error: str = ''


class Secretary:
    def __init__(self, settings, modem, audio, engines):
        self.settings = settings
        self.modem = modem
        self.audio = audio
        self.engines = engines
        self.call = None
        self.draining = None
        self.pending_caller = ('Unknown', 0.0)

    def _speak(self, call, text, stop_waiting=None):
        if call.cancelled.is_set():
            return False
        message_id = models.create_message(call.id, 'assistant', text, 'pending')
        try:
            waiting = (self.audio.thinking(call.cancelled) if stop_waiting is None
                       else nullcontext(stop_waiting))
            with waiting as stop:
                played = self.audio.play(self.engines.speech(text), self.engines.sample_rate,
                                         call.cancelled, stop_waiting=stop)
        except Exception:
            models.update_delivery(message_id, 'failed')
            raise
        models.update_delivery(message_id, 'played' if played else 'interrupted')
        if not played and not call.cancelled.is_set():
            raise RuntimeError('TTS produced no playable audio')
        return played

    def _conversation(self, call):
        try:
            if call.cancelled.wait(self.settings.greeting_delay):
                return
            if not self._speak(call, self.settings.greeting):
                return
            silent_turns = 0
            while not call.cancelled.is_set():
                text = self.audio.listen(call.cancelled).strip()
                if call.cancelled.is_set():
                    break
                if not text:
                    silent_turns += 1
                    if silent_turns >= self.settings.max_silent_turns:
                        call.reason = 'silence_timeout'
                        self._speak(call, self.settings.goodbye)
                        break
                    self._speak(call, self.settings.silence_message)
                    continue
                silent_turns = 0
                models.create_message(call.id, 'user', text)
                if is_goodbye(text):
                    call.reason = 'user_goodbye'
                    self._speak(call, self.settings.goodbye)
                    break  # tick() hangs up after playback finishes and done is set.
                with self.audio.thinking(call.cancelled) as stop_waiting:
                    reply = self.engines.reply(models.get_history(call.id))
                    if call.cancelled.is_set():
                        break
                    self._speak(call, reply, stop_waiting=stop_waiting)
        except Exception as exc:
            call.reason = 'processing_error'
            call.error = str(exc)
            log.exception('Call %s: processing failed', call.id)
        finally:
            call.done.set()

    def _finish(self, reason, error='', remote=False):
        call = self.call
        if call is None or call.ended:
            return
        call.cancelled.set()
        call.ended = True
        try:
            # Even on remote hangup, disable PCM before accepting another call.
            if remote:
                self.modem.command('AT+QPCMV=0')
            else:
                self.modem.hangup()
        except Exception as exc:
            error = f'{error}; {exc}'.strip('; ')
            # An unconfirmed hangup must not permit a new call on the same connection.
            self.modem.broken.set()
            log.error('Call %s cleanup failed: %s', call.id, exc)
        status = 'error' if error else ('completed' if call.answered_at is not None else 'missed')
        models.finish_call(call.id, status, reason, error)
        self.pending_caller = ('Unknown', 0.0)
        log.info('Call %s ended: %s', call.id, reason)

    def handle_event(self, event):
        current = time.monotonic()
        if event.kind == 'disconnected':
            if self.call and not self.call.ended:
                self.call.cancelled.set()
                self.call.ended = True
                models.finish_call(self.call.id, 'error', 'modem_disconnected', event.value)
            raise RuntimeError(f'Modem disconnected: {event.value}')
        if event.kind in ('ring', 'caller') and self.call and self.call.ended:
            if self.call.worker:
                if self.call.worker.is_alive():
                    self.draining = self.call
                else:
                    self.call.worker.join()
            self.call = None
        if event.kind == 'caller':
            if self.call is None:
                self.pending_caller = (event.value, current)
            elif not self.call.ended and self.call.answered_at is None:
                models.update_caller(self.call.id, event.value)
        elif event.kind == 'ring':
            if self.call is None:
                phone, received_at = self.pending_caller
                phone = phone if current - received_at < 5 else 'Unknown'
                self.pending_caller = ('Unknown', 0.0)
                self.call = Call(models.create_call(phone), current, current)
                log.info('Incoming call %s', self.call.id)
            elif not self.call.ended:
                self.call.last_ring = current
        elif event.kind == 'ended':
            self._finish('remote_hangup', remote=True)

    def tick(self):
        if self.draining and self.draining.done.is_set():
            self.draining.worker.join()
            self.draining = None
        call = self.call
        if call is None:
            return
        if call.ended:
            # A cancelled inference may still be running. Never share models between calls.
            if call.worker is None or call.done.is_set():
                if call.worker:
                    call.worker.join()
                self.call = None
            return
        current = time.monotonic()
        if call.answered_at is None:
            if current - call.last_ring >= self.settings.ring_timeout:
                self._finish('ring_timeout')
            elif self.draining is None and current - call.first_ring >= self.settings.answer_delay:
                try:
                    self.modem.answer()
                    call.answered_at = time.monotonic()
                    models.answer_call(call.id)
                    self.drain_events()
                    if call.ended:
                        return
                    call.worker = threading.Thread(target=self._conversation, args=(call,),
                                                   name=f'call-{call.id}', daemon=True)
                    call.worker.start()
                except Exception as exc:
                    self._finish('answer_failed', str(exc))
        elif call.done.is_set():
            self._finish(call.reason, call.error)
        elif current - call.answered_at >= self.settings.max_call_seconds:
            self._finish('max_duration')

    def drain_events(self):
        while True:
            try:
                event = self.modem.events.get_nowait()
            except queue.Empty:
                return
            self.handle_event(event)

    def run(self, stop):
        log.info('Ready. Waiting for incoming calls; Ctrl+C stops the service.')
        try:
            while not stop.is_set():
                try:
                    self.handle_event(self.modem.events.get(timeout=0.1))
                    # Process queued hangup/caller events before answering on a timer.
                    self.drain_events()
                except queue.Empty:
                    pass
                if self.modem.broken.is_set():
                    raise RuntimeError('Modem connection lost or cleanup failed; restart the service')
                if stop.is_set():
                    break
                self.tick()
        finally:
            self._finish('service_stopped')
            for call in (self.call, self.draining):
                if call and call.worker:
                    call.cancelled.set()
                    call.worker.join(timeout=2)

    def worker_alive(self):
        return any(call and call.worker and call.worker.is_alive() for call in (self.call, self.draining))
