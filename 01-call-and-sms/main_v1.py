"""Simple CLI controller for Quectel EC25 modem with call and SMS handling."""

import re
import shlex
import threading
import time
import serial
from models import CallLog, SMSMessage, init_db

# Serial port configuration
PORT: str = '/dev/ttyUSB2'
BAUDRATE: int = 115200


def decode_text(text: str) -> str:
    """Convert hex-encoded UCS2 text to readable Unicode."""
    clean_text = text.strip().replace(' ', '')
    try:
        return bytes.fromhex(clean_text).decode('utf-16-be')
    except Exception:
        return text


class Modem:
    """Wrapper to interact with the modem over serial port."""

    def __init__(self, port: str = PORT, baudrate: int = BAUDRATE) -> None:
        self.ser = serial.Serial(port, baudrate, timeout=0.5)
        self.lock = threading.Lock()
        self.running = True

        # Initial modem setup
        self.send('ATE0')  # Disable echo
        self.send('AT+CLIP=1')  # Enable caller ID presentation
        self.send('AT+CMGF=1')  # Set SMS to text mode
        self.send('AT+CSCS="GSM"')  # Default character set
        self.send('AT+CNMI=2,1,0,0,0')  # Direct new SMS notification

    def send(self, cmd: str, wait: float = 0.2) -> str:
        """Send an AT command and return response."""
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write(f'{cmd}\r\n'.encode('latin1'))
            time.sleep(wait)
            return self.ser.read_all().decode('latin1', errors='ignore').strip()

    def read_line(self) -> str:
        """Read a single line from serial buffer."""
        with self.lock:
            if self.ser.in_waiting > 0:
                return self.ser.readline().decode('latin1', errors='ignore').strip()
        return ''

    def get_status(self) -> dict[str, str]:
        """Fetch basic connection status."""
        signal = self.send('AT+CSQ')
        operator = self.send('AT+COPS?')
        net_type = self.send('AT+QNWINFO')

        return {
            'signal': signal.replace('\r\n', ' '),
            'operator': operator.replace('\r\n', ' '),
            'network': net_type.replace('\r\n', ' '),
        }

    def dial(self, phone: str) -> None:
        """Make an outgoing voice call."""
        self.send(f'ATD{phone};', wait=0.5)
        CallLog.create(phone_number=phone, call_type='outgoing', status='calling')

    def answer(self) -> None:
        """Answer an incoming call using ATA."""
        res = self.send('ATA', wait=0.5)
        if 'OK' in res or 'CONNECT' in res:
            print('Call connected.')
            CallLog.create(phone_number='Active Call', call_type='incoming', status='answered')
        else:
            print('Could not answer call.')

    def hangup(self) -> None:
        """End active call or reject incoming ring using ATH."""
        self.send('ATH', wait=0.2)
        print('Call disconnected.')

    def send_sms(self, phone: str, text: str) -> None:
        """Send an SMS message (auto-detects Persian/English)."""
        is_persian = any(ord(char) > 127 for char in text)

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
        """Read and save incoming SMS by index."""
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
        """Close modem connection."""
        self.running = False
        self.hangup()
        self.ser.close()


def listen_modem(modem: Modem) -> None:
    """Background listener for calls and SMS."""
    current_caller = 'Unknown'

    while modem.running:
        line = modem.read_line()
        if not line:
            time.sleep(0.1)
            continue

        # Extract caller ID
        if '+CLIP:' in line:
            match = re.search(r'\+CLIP:\s*"([^"]+)"', line)
            if match:
                current_caller = match.group(1)

        # Incoming call ring
        elif 'RING' in line:
            print(
                f'\n[Incoming Call: {current_caller}] Type "answer" or "hangup"\n> ',
                end='',
                flush=True,
            )
            CallLog.create(phone_number=current_caller, call_type='incoming', status='ringing')

        # Call ended by remote side
        elif 'NO CARRIER' in line or 'BUSY' in line:
            print(f'\n[Call Ended: {line}]\n> ', end='', flush=True)
            current_caller = 'Unknown'

        # New SMS received
        elif '+CMTI:' in line:
            match = re.search(r',(\d+)', line)
            if match:
                modem.read_sms(int(match.group(1)))

        time.sleep(0.1)


def show_help() -> None:
    """Display available commands."""
    print(
        '\nCommands:\n'
        '  status               - Check signal and network\n'
        '  call <number>        - Dial a phone number\n'
        '  answer               - Answer incoming call (ATA)\n'
        '  hangup               - End or reject call (ATH)\n'
        '  sms <number> <text>  - Send SMS message\n'
        '  logs                 - Show recent call and SMS logs\n'
        '  help                 - Show command list\n'
        '  exit                 - Quit program\n'
    )


def show_logs() -> None:
    """Print saved logs from database."""
    print('\n--- Calls ---')
    for call in CallLog.select().order_by(CallLog.timestamp.desc()).limit(5):
        print(f'{call.timestamp} | {call.call_type} | {call.phone_number} [{call.status}]')

    print('\n--- SMS ---')
    for sms in SMSMessage.select().order_by(SMSMessage.timestamp.desc()).limit(5):
        print(f'{sms.timestamp} | {sms.direction} | {sms.phone_number}: {sms.message}')
    print()


def main() -> None:
    init_db()
    modem = Modem(PORT, BAUDRATE)

    print('\nChecking modem status...')
    for key, value in modem.get_status().items():
        print(f'{key}: {value}')

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
                for key, value in modem.get_status().items():
                    print(f'{key}: {value}')
            elif cmd == 'call' and len(parts) > 1:
                modem.dial(parts[1])
                print(f'Calling {parts[1]}...')
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
        print('\nExited.')


if __name__ == '__main__':
    main()