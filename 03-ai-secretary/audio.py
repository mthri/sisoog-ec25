"""EC25 PCM in/out only. The host microphone/speakers are never selected implicitly."""
import queue
import threading
import time
from collections import deque
from contextlib import contextmanager
from math import gcd

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

RATE = 8000
FRAME = 256  # Silero supports 32 ms at 8 kHz.


def resample(audio, source_rate: int, target_rate: int):
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_rate == target_rate:
        return waveform
    factor = gcd(source_rate, target_rate)
    return resample_poly(waveform, target_rate // factor, source_rate // factor).astype(np.float32)


def select_devices(input_device=None, output_device=None):
    """Prefer native modem hardware; explicit indices override each direction."""
    devices = sd.query_devices()
    for direction, chosen in (('input', input_device), ('output', output_device)):
        if chosen is not None:
            if chosen < 0 or chosen >= len(devices) or devices[chosen][f'max_{direction}_channels'] < 1:
                raise ValueError(f'Invalid {direction} device: {chosen}')
    resolved = []
    for direction, chosen in (('input', input_device), ('output', output_device)):
        if chosen is None:
            ranked = []
            for i, device in enumerate(devices):
                name = device['name'].lower()
                if device[f'max_{direction}_channels'] < 1 or 'monitor' in name:
                    continue
                branded = any(tag in name for tag in ('ec25', 'quectel'))
                if branded or 'usb audio' in name:
                    # Prefer ALSA hardware over duplicate sound-server aliases.
                    ranked.append(((branded, '(hw:' in name), i))
            best = max((rank for rank, _ in ranked), default=None)
            candidates = [i for rank, i in ranked if rank == best]
            if len(candidates) != 1:
                matches = ', '.join(f'{i}: {devices[i]["name"]}' for i in candidates) or 'none'
                raise RuntimeError(f'Cannot uniquely identify modem {direction}. '
                                   f'Candidates: {matches}. '
                                   f'Use --list-devices and --{direction}-device INDEX.')
            chosen = candidates[0]
        resolved.append(chosen)
    for direction, chosen, check in (
        ('input', resolved[0], sd.check_input_settings),
        ('output', resolved[1], sd.check_output_settings),
    ):
        try:
            check(device=chosen, channels=1, dtype='int16', samplerate=RATE)
        except sd.PortAudioError as exc:
            raise RuntimeError(
                f'{direction} device {chosen} ({devices[chosen]["name"]}) cannot use '
                f'{RATE} Hz mono int16: {exc}. Select the modem with '
                f'--{direction}-device INDEX, or omit that option for automatic detection. '
                'Use --list-devices to see current indices.'
            ) from exc
    return tuple(resolved)


class PhoneAudio:
    def __init__(self, input_device, output_device, asr, settings):
        self.input_device = input_device
        self.output_device = output_device
        self.asr = asr
        self.settings = settings

    @contextmanager
    def thinking(self, cancelled):
        """Play a quiet double beep until stopped, cancelled, or the scope exits."""
        stopped = threading.Event()
        errors = []

        def run():
            stream = None
            try:
                # Fast/cached replies should not make an unnecessary beep.
                deadline = time.monotonic() + 0.4
                while time.monotonic() < deadline:
                    if stopped.wait(0.02) or cancelled.is_set():
                        return
                cycle = np.zeros(RATE * 2, dtype=np.int16)
                count = int(RATE * 0.1)
                envelope = np.ones(count)
                fade = int(RATE * 0.01)
                envelope[:fade] = np.linspace(0, 1, fade)
                envelope[-fade:] = np.linspace(1, 0, fade)
                for offset, frequency in ((0, 660), (int(RATE * 0.2), 880)):
                    wave = np.sin(2 * np.pi * frequency * np.arange(count) / RATE)
                    cycle[offset:offset + count] = (
                        wave * envelope * self.settings.thinking_volume * 32767
                    ).astype(np.int16)
                stream = sd.OutputStream(device=self.output_device, samplerate=RATE,
                                         channels=1, dtype='int16', latency='low')
                stream.start()
                position = 0
                while not stopped.is_set() and not cancelled.is_set():
                    stream.write(cycle[position:position + 320].reshape(-1, 1))
                    position = (position + 320) % len(cycle)
            except Exception as exc:
                errors.append(exc)
            finally:
                if stream is not None:
                    try:
                        try:
                            stream.abort()
                        finally:
                            stream.close()
                    except Exception as exc:
                        errors.append(exc)

        worker = None

        def stop():
            stopped.set()
            if worker is not None:
                worker.join()  # Release the modem output before speech opens it.
            if errors:
                raise errors[0]

        if self.settings.thinking_enabled and not cancelled.is_set():
            if not 0 <= self.settings.thinking_volume <= 1:
                raise ValueError('thinking_volume must be between 0 and 1')
            if self.settings.thinking_volume > 0:
                worker = threading.Thread(target=run, name='phone-thinking', daemon=True)
                worker.start()
        try:
            yield stop
        finally:
            stop()

    def listen(self, cancelled):
        """Capture one bounded utterance, then close input before running ASR."""
        import torch
        chunks = queue.Queue(maxsize=128)
        overflow = []

        def capture(indata, frames, timing, status):
            if cancelled.is_set():
                raise sd.CallbackStop
            if status.input_overflow:
                overflow.append('Audio input overflow')
            try:
                chunks.put_nowait(indata[:, 0].astype(np.float32) / 32768.0)
            except queue.Full:
                overflow.append('Speech processing cannot keep up with input')

        vad = self.asr.vad_model
        vad.reset_states()
        started = time.monotonic()
        speech_started = None
        silence = speech_samples = 0
        captured = []
        preroll = deque(maxlen=6)
        with sd.InputStream(device=self.input_device, samplerate=RATE, channels=1,
                            dtype='int16', blocksize=FRAME, callback=capture):
            while not cancelled.is_set():
                if overflow:
                    raise RuntimeError(overflow[0])
                current = time.monotonic()
                if speech_started is None and current - started >= self.settings.silence_timeout:
                    break
                if speech_started is not None and current - speech_started >= self.settings.phrase_limit:
                    break
                try:
                    chunk = chunks.get(timeout=0.1)
                except queue.Empty:
                    continue
                with torch.inference_mode():
                    probability = vad(torch.from_numpy(chunk), RATE).item()
                speech = probability > self.asr.config.vad_threshold
                if speech and speech_started is None:
                    speech_started = current
                    captured.extend(preroll)
                if speech_started is not None:
                    captured.append(chunk)
                    speech_samples += len(chunk) if speech else 0
                    silence = 0 if speech else silence + len(chunk)
                    if silence >= RATE * self.asr.config.end_silence_ms / 1000:
                        break
                else:
                    preroll.append(chunk)
        if cancelled.is_set() or speech_samples < RATE * self.asr.config.min_speech_ms / 1000:
            return ''
        return self.asr.transcribe(resample(np.concatenate(captured), RATE, self.asr.config.sample_rate))

    def play(self, segments, source_rate, cancelled, stop_waiting=None):
        """Prepare validated PCM ahead while the caller plays the preceding chunks."""
        if cancelled.is_set():
            return False
        ready = queue.Queue(maxsize=3)
        stopped = threading.Event()
        errors = []

        def enqueue(value):
            while not stopped.is_set() and not cancelled.is_set():
                try:
                    ready.put(value, timeout=0.1)
                    return
                except queue.Full:
                    continue

        def produce():
            iterator = None
            try:
                iterator = iter(segments)
                while not stopped.is_set() and not cancelled.is_set():
                    try:
                        segment = next(iterator)
                    except StopIteration:
                        break
                    if stopped.is_set() or cancelled.is_set():
                        break
                    waveform = resample(segment, source_rate, RATE)
                    if len(waveform):
                        enqueue((np.clip(waveform, -1, 1) * 32767).astype(np.int16))
            except Exception as exc:
                errors.append(exc)
            finally:
                try:
                    if hasattr(iterator, 'close'):
                        iterator.close()
                except Exception as exc:
                    errors.append(exc)
                enqueue(None)

        worker = threading.Thread(target=produce, name='phone-tts-prefetch', daemon=True)
        stream = None
        worker.start()
        try:
            while not cancelled.is_set():
                try:
                    pcm = ready.get(timeout=0.1)
                except queue.Empty:
                    continue
                if pcm is None:
                    if errors:
                        raise errors[0]
                    break
                if cancelled.is_set():
                    return False
                if stream is None:
                    if stop_waiting is not None:
                        stop_waiting()
                    if cancelled.is_set():
                        return False
                    stream = sd.OutputStream(device=self.output_device, samplerate=RATE,
                                             channels=1, dtype='int16', latency='low')
                    stream.start()
                for start in range(0, len(pcm), 320):
                    if cancelled.is_set():
                        return False
                    stream.write(pcm[start:start + 320].reshape(-1, 1))
            if stream is not None and not cancelled.is_set():
                stream.stop()  # Drain the last samples before marking the message played.
            return stream is not None and not cancelled.is_set()
        finally:
            stopped.set()
            try:
                if stream is not None:
                    try:
                        stream.abort()
                    finally:
                        stream.close()
            finally:
                # A running inference cannot be interrupted. Wait before ASR/model reuse,
                # but release the audio device immediately on hangup or playback failure.
                worker.join()
