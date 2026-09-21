"""Single serial reader: AT responses never consume or erase call notifications."""
import queue
import re
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    kind: str
    value: str = ''


class Modem:
    def __init__(self, port: str, baudrate: int, transport=None):
        if transport is None:
            import serial
            transport = serial.Serial(port, baudrate, timeout=0.1, write_timeout=2, exclusive=True)
        self.serial = transport
        self.events = queue.Queue()
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.pending = None
        self.pending_command = ''
        self.stopped = threading.Event()
        self.broken = threading.Event()
        self.reader = threading.Thread(target=self._read, name='modem-reader', daemon=True)
        self.reader.start()

    def _read(self):
        buffer = b''
        try:
            while not self.stopped.is_set():
                chunk = self.serial.read(256)
                if not chunk:
                    continue
                buffer += chunk
                while b'\n' in buffer:
                    raw, buffer = buffer.split(b'\n', 1)
                    line = raw.decode('ascii', errors='replace').strip()
                    if line:
                        self._line(line)
                if len(buffer) > 65536:
                    raise RuntimeError('Serial input has no line terminator')
        except Exception as exc:
            if not self.stopped.is_set():
                self.broken.set()
                self.events.put(Event('disconnected', str(exc)))
                with self.state_lock:
                    if self.pending is not None:
                        self.pending.put('ERROR: serial disconnected')

    def _line(self, line: str):
        if line == 'RING' or line.startswith('+CRING:'):
            self.events.put(Event('ring'))
            return
        if line.startswith('+CLIP:'):
            match = re.search(r'\+CLIP:\s*"([^"]*)"', line)
            if match:
                self.events.put(Event('caller', match.group(1) or 'Unknown'))
            return
        if line in ('NO CARRIER', 'BUSY', 'NO ANSWER'):
            self.events.put(Event('ended', line))
            with self.state_lock:
                if self.pending is not None and self.pending_command == 'ATA':
                    self.pending.put(line)
            return
        with self.state_lock:
            if self.pending is not None:
                self.pending.put(line)

    def command(self, command: str, timeout: float = 5) -> list[str]:
        with self.lock:
            if self.broken.is_set() or self.stopped.is_set():
                raise RuntimeError('Modem connection is unavailable; restart the service')
            response = queue.Queue()
            with self.state_lock:
                self.pending = response
                self.pending_command = command
            try:
                self.serial.write((command + '\r\n').encode('ascii'))
                deadline = time.monotonic() + timeout
                lines = []
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    line = response.get(timeout=remaining)
                    if line == 'OK' or line == 'CONNECT' or line.startswith('CONNECT '):
                        return lines
                    if line.startswith(('ERROR', '+CME ERROR:', '+CMS ERROR:')) or line in ('NO CARRIER', 'BUSY', 'NO ANSWER'):
                        raise RuntimeError(f'{command}: {line}')
                    lines.append(line)
            except queue.Empty as exc:
                # A late OK cannot safely be assigned to a future command.
                self.broken.set()
                self.events.put(Event('disconnected', f'Timeout: {command}'))
                raise TimeoutError(f'Modem did not answer {command}') from exc
            except OSError:
                self.broken.set()
                self.events.put(Event('disconnected', 'Serial write failed'))
                raise
            finally:
                with self.state_lock:
                    self.pending = None
                    self.pending_command = ''

    def initialize(self):
        for command in ('AT', 'ATE0', 'AT+CLIP=1'):
            self.command(command)

    def answer(self):
        self.command('ATA', timeout=10)
        self.command('AT+QPCMV=1,2')

    def hangup(self):
        errors = []
        for command in ('ATH', 'AT+QPCMV=0'):
            try:
                self.command(command)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('; '.join(errors))

    def close(self):
        self.stopped.set()
        self.reader.join(timeout=1)
        self.serial.close()
