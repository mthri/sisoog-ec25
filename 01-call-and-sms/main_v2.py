"""CLI modem controller for Quectel EC25 with call audio bridge and Persian SMS."""

import queue
import re
import shlex
import threading
import time
import serial
import sounddevice as sd
from models import CallLog, SMSMessage, init_db

PORT: str = '/dev/ttyUSB0'
BAUDRATE: int = 115200
SAMPLE_RATE: int = 8000
BLOCK_SIZE: int = 320


def decode_text(text: str) -> str:
    """Convert hex-encoded UCS2 text to readable Unicode."""
    clean_text = text.strip().replace(' ', '')
    try:
        return bytes.fromhex(clean_text).decode('utf-16-be')
    except Exception:
        return text


def detect_audio_devices() -> tuple[int, int, int, int] | None:
    """Detect audio device indices for EC25 and host sound card."""
    devices = sd.query_devices()
    modem_in: int | None = None
    modem_out: int | None = None

    for idx, dev in enumerate(devices):
        name = str(dev['name']).lower()
        if 'ec25' in name or 'usb audio' in name:
            if dev['max_input_channels'] > 0 and modem_in is None:
                modem_in = idx
            if dev['max_output_channels'] > 0 and modem_out is None:
                modem_out = idx

    if modem_in is None or modem_out is None:
        return None

    laptop_in = sd.default.device[0]
    laptop_out = sd.default.device[1]
    return laptop_in, modem_out, modem_in, laptop_out


class AudioBridge:
    """Handles bidirectional PCM streaming between host and modem."""

    def __init__(self, devices: tuple[int, int, int, int]) -> None:
        self.laptop_in, self.modem_out, self.modem_in, self.laptop_out = devices
        self.q_tx: queue.Queue = queue.Queue(maxsize=30)
        self.q_rx: queue.Queue = queue.Queue(maxsize=30)
        self.active: bool = False
        self.streams: list[sd._Stream] = []

    def _flush_queues(self) -> None:
        for q in (self.q_tx, self.q_rx):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def start(self) -> None:
        """Start mic capture and speaker playback streams."""
        if self.active:
            return
        self._flush_queues()
        self.active = True

        def laptop_mic_cb(indata, frames, time_info, status):
            if self.active:
                try:
                    self.q_tx.put_nowait(indata.copy())
                except queue.Full:
                    pass

        def modem_tx_cb(outdata, frames, time_info, status):
            try:
                outdata[:] = self.q_tx.get_nowait()
            except queue.Empty:
                outdata.fill(0)

        def modem_rx_cb(indata, frames, time_info, status):
            if self.active:
                try:
                    self.q_rx.put_nowait(indata.copy())
                except queue.Full:
                    pass

        def laptop_spk_cb(outdata, frames, time_info, status):
            try:
                outdata[:] = self.q_rx.get_nowait()
            except queue.Empty:
                outdata.fill(0)

        self.streams = [
            sd.InputStream(
                device=self.laptop_in,
                channels=1,
                samplerate=SAMPLE_RATE,
                dtype='int16',
                blocksize=BLOCK_SIZE,
                callback=laptop_mic_cb,
            ),
            sd.OutputStream(
                device=self.modem_out,
                channels=1,
                samplerate=SAMPLE_RATE,
                dtype='int16',
                blocksize=BLOCK_SIZE,
                callback=modem_tx_cb,
            ),
            sd.InputStream(
                device=self.modem_in,
                channels=1,
                samplerate=SAMPLE_RATE,
                dtype='int16',
                blocksize=BLOCK_SIZE,
                callback=modem_rx_cb,
            ),
            sd.OutputStream(
                device=self.laptop_out,
                channels=1,
                samplerate=SAMPLE_RATE,
                dtype='int16',
                blocksize=BLOCK_SIZE,
                callback=laptop_spk_cb,
            ),
        ]

        for s in self.streams:
            s.start()
        print('[Audio] Audio bridge active.')

    def stop(self) -> None:
        """Stop all audio streams and release ALSA handles."""
        if not self.active:
            return
        self.active = False
        for s in self.streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self.streams.clear()
        self._flush_queues()
        print('[Audio] Audio bridge stopped.')


class Modem:
    """Modem controller for voice, SMS, and network operations."""

    def __init__(
        self,
        port: str = PORT,
        baudrate: int = BAUDRATE,
        audio_bridge: AudioBridge | None = None,
    ) -> None:
        self.ser = serial.Serial(port, baudrate, timeout=0.5)
        self.lock = threading.Lock()
        self.running: bool = True
        self.audio: AudioBridge | None = audio_bridge
        self.in_call: bool = False

        self.send('ATE0')
        self.send('AT+CLIP=1')
        self.send('AT+CMGF=1')
        self.send('AT+CSCS="GSM"')
        self.send('AT+CNMI=2,1,0,0,0')

    def send(self, cmd: str, wait: float = 0.2) -> str:
        """Send raw AT command and read response."""
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write(f'{cmd}\r\n'.encode('latin1'))
            time.sleep(wait)
            return self.ser.read_all().decode('latin1', errors='ignore').strip()

    def read_line(self) -> str:
        """Read a single line from the serial buffer."""
        with self.lock:
            if self.ser.in_waiting > 0:
                return self.ser.readline().decode('latin1', errors='ignore').strip()
        return ''

    def get_status(self) -> dict[str, str]:
        """Query signal quality and registration status."""
        return {
            'signal': self.send('AT+CSQ').replace('\r\n', ' '),
            'operator': self.send('AT+COPS?').replace('\r\n', ' '),
            'network': self.send('AT+QNWINFO').replace('\r\n', ' '),
        }

    def dial(self, phone: str) -> None:
        """Dial outgoing number and start audio routing."""
        res = self.send(f'ATD{phone};', wait=0.5)
        if 'OK' in res:
            self.in_call = True
            self.send('AT+QPCMV=1,2', wait=0.1)
            if self.audio:
                self.audio.start()
            CallLog.create(phone_number=phone, call_type='outgoing', status='calling')
            print(f'Calling {phone}...')
        else:
            print('Failed to dial.')

    def answer(self) -> None:
        """Answer incoming call and bridge voice lines."""
        res = self.send('ATA', wait=0.5)
        if 'OK' in res or 'CONNECT' in res:
            self.in_call = True
            self.send('AT+QPCMV=1,2', wait=0.1)
            if self.audio:
                self.audio.start()
            CallLog.create(phone_number='Active Call', call_type='incoming', status='answered')
            print('Call connected.')
        else:
            print('Failed to answer.')

    def hangup(self) -> None:
        """End voice call and tear down audio bridge."""
        if self.audio:
            self.audio.stop()
        self.send('AT+QPCMV=0', wait=0.1)
        self.send('ATH', wait=0.2)
        self.in_call = False
        print('Call ended.')

    def send_sms(self, phone: str, text: str) -> None:
        """Send plain text or Persian UCS-2 SMS."""
        is_persian = any(ord(c) > 127 for c in text)

        with self.lock:
            if is_persian:
                self.ser.write(b'AT+CSCS="UCS2"\r\n')
                time.sleep(0.1)
                self.ser.write(b'AT+CSMP=17,167,0,8\r\n')
                time.sleep(0.1)
                target_phone = ''.join(f'{ord(c):04X}' for c in phone)
                payload = ''.join(f'{ord(c):04X}' for c in text)
            else:
                target_phone = phone
                payload = text

            self.ser.reset_input_buffer()
            self.ser.write(f'AT+CMGS="{target_phone}"\r\n'.encode('latin1'))
            time.sleep(0.3)
            self.ser.write(f'{payload}\x1a'.encode('latin1'))
            time.sleep(2.0)

            if is_persian:
                self.ser.write(b'AT+CSCS="GSM"\r\n')
                time.sleep(0.1)
                self.ser.write(b'AT+CSMP=17,167,0,0\r\n')

        SMSMessage.create(
            phone_number=phone,
            message=text,
            direction='outbound',
            status='sent',
        )

    def read_sms(self, index: int) -> None:
        """Fetch incoming SMS from SIM storage and decode."""
        raw = self.send(f'AT+CMGR={index}', wait=0.5)
        lines = [line.strip() for line in raw.splitlines() if line.strip()]

        sender = 'Unknown'
        body = ''

        for idx, line in enumerate(lines):
            if line.startswith('+CMGR:'):
                match = re.search(r'"([^"]+)"', line)
                if match:
                    sender = match.group(1)
                if idx + 1 < len(lines):
                    body = lines[idx + 1]
                break

        body = decode_text(body)
        SMSMessage.create(
            phone_number=sender,
            message=body,
            direction='inbound',
            status='received',
        )
        print(f'\n[New SMS] From: {sender} -> {body}\n> ', end='', flush=True)
        self.send(f'AT+CMGD={index}', wait=0.1)

    def close(self) -> None:
        """Shut down modem controller and release resources."""
        self.running = False
        self.hangup()
        self.ser.close()


def listen_modem(modem: Modem) -> None:
    """Background listener for URC notifications."""
    current_caller = 'Unknown'

    while modem.running:
        line = modem.read_line()
        if not line:
            time.sleep(0.1)
            continue

        if '+CLIP:' in line:
            match = re.search(r'\+CLIP:\s*"([^"]+)"', line)
            if match:
                current_caller = match.group(1)

        elif 'RING' in line:
            print(
                f'\n[Incoming Call: {current_caller}] Type "answer" or "hangup"\n> ',
                end='',
                flush=True,
            )
            CallLog.create(phone_number=current_caller, call_type='incoming', status='ringing')

        elif 'NO CARRIER' in line or 'BUSY' in line:
            print(f'\n[Remote Call Terminated: {line}]\n> ', end='', flush=True)
            if modem.in_call:
                modem.hangup()
            current_caller = 'Unknown'

        elif '+CMTI:' in line:
            match = re.search(r',(\d+)', line)
            if match:
                modem.read_sms(int(match.group(1)))

        time.sleep(0.1)


def show_help() -> None:
    """Print available CLI commands."""
    print(
        '\nCommands:\n'
        '  status               - Check signal and network info\n'
        '  call <number>        - Dial number with mic/speaker active\n'
        '  answer               - Answer incoming call and route audio\n'
        '  hangup               - End active call and disconnect audio\n'
        '  sms <number> <text>  - Send SMS (English or Persian)\n'
        '  logs                 - Show recent calls and messages\n'
        '  help                 - Display this menu\n'
        '  exit                 - Quit application\n'
    )


def show_logs() -> None:
    """Display latest database entries."""
    print('\n--- Recent Calls ---')
    for call in CallLog.select().order_by(CallLog.timestamp.desc()).limit(5):
        print(f'{call.timestamp} | {call.call_type} | {call.phone_number} [{call.status}]')

    print('\n--- Recent SMS ---')
    for sms in SMSMessage.select().order_by(SMSMessage.timestamp.desc()).limit(5):
        print(f'{sms.timestamp} | {sms.direction} | {sms.phone_number}: {sms.message}')
    print()


def main() -> None:
    init_db()

    # Detect sound cards for EC25 and host
    audio_devices = detect_audio_devices()
    audio_bridge = AudioBridge(audio_devices) if audio_devices else None

    if audio_bridge:
        print('[System] Audio hardware detected. Voice calls enabled.')
    else:
        print('[Warning] EC25 sound card not found. Audio bridging unavailable.')

    modem = Modem(PORT, BAUDRATE, audio_bridge=audio_bridge)

    print('\nModem Status:')
    for key, val in modem.get_status().items():
        print(f'  {key}: {val}')

    show_help()

    thread = threading.Thread(target=listen_modem, args=(modem,), daemon=True)
    thread.start()

    try:
        while True:
            cmd_input = input('> ').strip()
            if not cmd_input:
                continue

            parts = shlex.split(cmd_input)
            cmd = parts[0].lower()

            if cmd in ('exit', 'quit'):
                break
            elif cmd == 'help':
                show_help()
            elif cmd == 'status':
                for key, val in modem.get_status().items():
                    print(f'  {key}: {val}')
            elif cmd == 'call' and len(parts) > 1:
                modem.dial(parts[1])
            elif cmd == 'answer':
                modem.answer()
            elif cmd == 'hangup':
                modem.hangup()
            elif cmd == 'sms' and len(parts) > 2:
                phone = parts[1]
                msg = ' '.join(parts[2:])
                modem.send_sms(phone, msg)
                print('SMS sent.')
            elif cmd == 'logs':
                show_logs()
            else:
                print('Unknown command. Type "help".')

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        modem.close()
        print('\nTerminated.')


if __name__ == '__main__':
    main()