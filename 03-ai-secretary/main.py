"""Run from any directory using the existing 02-ai virtual environment."""
import argparse
import json
import logging
import signal
import threading
from dataclasses import replace
from pathlib import Path

from config import Settings


def positive(value):
    number = float(value)
    if not 0 < number < float('inf'):
        raise argparse.ArgumentTypeError('must be a positive finite number')
    return number


def parser():
    defaults = Settings()
    p = argparse.ArgumentParser(description='Local Persian telephone secretary for Quectel EC25')
    p.add_argument('--port', default=defaults.port)
    p.add_argument('--baudrate', type=int, default=defaults.baudrate)
    p.add_argument('--input-device', type=int)
    p.add_argument('--output-device', type=int)
    p.add_argument('--database', type=Path, default=defaults.database)
    p.add_argument('--prompt-file', type=Path, default=defaults.prompt_file)
    p.add_argument('--asr-model', type=Path, default=defaults.asr_model)
    p.add_argument('--tts-dir', type=Path, default=defaults.tts_dir)
    p.add_argument('--voice', type=Path, default=defaults.voice)
    p.add_argument('--greeting-file', type=Path, default=defaults.greeting_file)
    p.add_argument('--llm-url', default=defaults.llm_url)
    p.add_argument('--llm-model', default=defaults.llm_model)
    for name in ('request-timeout', 'answer-delay', 'silence-timeout', 'max-call-seconds', 'phrase-limit', 'voice-prompt-seconds'):
        p.add_argument(f'--{name}', type=positive, default=getattr(defaults, name.replace('-', '_')))
    p.add_argument('--min-rms', type=positive, default=defaults.min_rms)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--list-devices', action='store_true', help='List audio devices, without loading models')
    mode.add_argument('--check', action='store_true', help='Check model files and audio formats, without opening serial')
    mode.add_argument('--logs', action='store_true', help='Show recent calls as JSON')
    mode.add_argument('--transcript', type=int, metavar='CALL_ID', help='Show a call transcript as JSON')
    return p


def check_files(settings):
    for path in (settings.asr_model, settings.voice, settings.prompt_file,
                 settings.tts_dir / 'farsi.yaml', settings.tts_dir / 'model.safetensors',
                 settings.tts_dir / 'tokenizer.model'):
        if not path.is_file():
            raise FileNotFoundError(f'Missing required file: {path}')


def main():
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    fields = Settings.__dataclass_fields__
    settings = replace(Settings(), **{key: value for key, value in vars(args).items() if key in fields})
    try:
        if args.logs or args.transcript is not None:
            import models
            models.init_db(settings.database)
            data = models.get_calls() if args.logs else models.get_transcript(args.transcript)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return
        from audio import PhoneAudio, select_devices
        if args.list_devices:
            import sounddevice as sd
            print(sd.query_devices())
            return
        check_files(settings)
        devices = select_devices(settings.input_device, settings.output_device)
        import sounddevice as sd
        print(f'[Audio] input={devices[0]} ({sd.query_devices(devices[0])["name"]}), '
              f'output={devices[1]} ({sd.query_devices(devices[1])["name"]}), 8000 Hz mono PCM')
        if args.check:
            print('Model files and audio formats OK. Serial, Ollama and model inference were not tested.')
            return
        # Serialize database recovery; pyserial separately locks the serial device.
        import fcntl
        settings.database.parent.mkdir(parents=True, exist_ok=True)
        with settings.database.with_suffix('.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Another secretary is already using this database') from exc
            from engines import Engines
            from modem import Modem
            from secretary import Secretary
            import models
            models.init_db(settings.database)
            models.recover_calls()
            engines = Engines(settings)
            modem = service = None
            stop = threading.Event()
            old_signals = {}
            try:
                for signum in (signal.SIGINT, signal.SIGTERM):
                    old_signals[signum] = signal.signal(signum, lambda *_: stop.set())
                modem = Modem(settings.port, settings.baudrate)
                modem.initialize()
                audio = PhoneAudio(*devices, engines.asr, settings)
                service = Secretary(settings, modem, audio, engines)
                service.run(stop)
            finally:
                if modem:
                    modem.close()
                if service is None or not service.worker_alive():
                    engines.close()
                for signum, handler in old_signals.items():
                    signal.signal(signum, handler)
    except KeyboardInterrupt:
        print('\nStopped.')
    except Exception as exc:
        raise SystemExit(f'Secretary failed: {exc}') from exc


if __name__ == '__main__':
    main()
