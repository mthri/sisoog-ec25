from __future__ import annotations

import queue
import re
import sys
import tempfile
import threading
import wave
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import sounddevice as sd
import sphn
import yaml
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.config import Config

# Add pocket-tts directory to sys.path if needed to load normalize_fa
_POCKET_DIR = Path(__file__).resolve().parent / 'pocket-tts'
if str(_POCKET_DIR) not in sys.path:
    sys.path.insert(0, str(_POCKET_DIR))

try:
    from normalize_fa import normalize as normalize_persian
except ImportError:
    normalize_persian = None


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PocketTTSConfig:
    """Configuration parameters for real-time Pocket TTS speech synthesis."""

    model_dir: Path | str = _POCKET_DIR
    voice_path: Path | str = _POCKET_DIR / 'example_voice.wav'
    sample_rate: int = 24_000
    temperature: float = 0.3
    eos_threshold: float = -2.0
    max_tokens: int = 18
    min_tokens: int = 8
    chunk_pause_seconds: float = 0.15
    join_seconds: float = 0.15
    pause_at_punct: bool = False
    voice_prompt_max_sec: float = 5.0
    frames_after_eos: int = 0
    use_cuda: bool = False


# Short name used by main.py; both names refer to the same configuration.
TTSConfig = PocketTTSConfig


# ============================================================
# TEXT PROCESSING & NORMALIZATION
# ============================================================

HARD_BREAK: Final = re.compile(r'(?<=[.!؟])\s+')
SOFT_BREAK: Final = re.compile(r'(?<=[،؛:])\s+')
BREAK_CHARS: Final[tuple[str, ...]] = ('.', '!', '؟', '،', '؛', ':')

FALLBACK_REPLACEMENTS: Final[tuple[tuple[str, str], ...]] = (
    ('...', '. '),
    ('…', '. '),
    (':', '، '),
    (' - ', '، '),
    (';', '، '),
    ('—', '-'),
    ('–', '-'),
    ('“', "'"),
    ('”', "'"),
    ('‘', "'"),
    ('’', "'"),
)


def has_spoken_text(text: str) -> bool:
    """Punctuation alone is not a valid speech prompt; keep letters and numbers."""
    return any(char.isalnum() for char in text)


def normalize_text(text: str) -> str:
    """Normalize Persian characters, numerals, and punctuation."""
    if normalize_persian is not None:
        return normalize_persian(text)

    # Fallback normalization if normalize_fa module is unavailable
    text = ' '.join(text.split()).strip()
    text = text.replace('آ', 'ا\u0653')
    text = text.replace('أ', 'ا').replace('إ', 'ا')
    text = re.sub(r'(?<![\u200c\w])ای\b', 'ا\u0650ی', text)
    text = text.replace('ي', 'ی').replace('ك', 'ک')
    for old, new in FALLBACK_REPLACEMENTS:
        text = text.replace(old, new)
    return ' '.join(text.split()).strip()


def split_text(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 18,
    keep_punct_boundaries: bool = False,
    min_tokens: int = 8,
) -> list[str]:
    """Split text into chunks of at most max_tokens respecting punctuation and boundaries."""
    text = ' '.join(text.split())
    if not text:
        return []

    def fits(s: str) -> bool:
        return count_tokens(s) <= max_tokens

    def split_by(pattern: re.Pattern, piece: str) -> list[str]:
        parts = [part.strip() for part in pattern.split(piece) if part.strip()]
        return parts if len(parts) > 1 else []

    def recurse(piece: str) -> list[str]:
        if fits(piece):
            return [piece]

        for pattern in (HARD_BREAK, SOFT_BREAK):
            parts = split_by(pattern, piece)
            if parts:
                return [chunk for part in parts for chunk in recurse(part)]

        output: list[str] = []
        current = ''
        for word in piece.split():
            trial = f'{current} {word}'.strip() if current else word
            if current and not fits(trial):
                output.append(current)
                current = word
            else:
                current = trial
        if current:
            output.append(current)
        return output

    if keep_punct_boundaries:
        pieces = [text]
        for pattern in (HARD_BREAK, SOFT_BREAK):
            pieces = [chunk for piece in pieces for chunk in (split_by(pattern, piece) or [piece])]
        parts = [chunk for piece in pieces for chunk in recurse(piece)]
    else:
        parts = recurse(text)

    # Merge neighboring chunks whenever possible
    merged: list[str] = []
    for chunk in parts:
        keep_apart = (
            keep_punct_boundaries
            and merged
            and merged[-1].rstrip().endswith(BREAK_CHARS)
            and count_tokens(merged[-1]) >= min_tokens
        )
        if merged and not keep_apart and fits(f'{merged[-1]} {chunk}'):
            merged[-1] = f'{merged[-1]} {chunk}'
        else:
            merged.append(chunk)

    return merged


# ============================================================
# AUDIO UTILITIES
# ============================================================

def trim_silence(
    audio: np.ndarray,
    sample_rate: int = 24_000,
    threshold_db: float = -38.0,
) -> np.ndarray:
    """Trim trailing dead air from audio buffer based on RMS energy."""
    chunk_size = int(sample_rate * 0.02)
    if len(audio) < chunk_size:
        return audio

    threshold = 10.0 ** (threshold_db / 20.0)
    end = len(audio)

    while end >= chunk_size:
        rms = float(np.sqrt(np.mean(audio[end - chunk_size:end] ** 2)))
        if rms > threshold:
            break
        end -= chunk_size

    margin = int(sample_rate * 0.06)
    cutoff = min(end + margin, len(audio))
    return audio[:cutoff]


def save_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    """Serialize audio array into a 16-bit mono WAV file."""
    clipped = np.clip(waveform.reshape(-1), -0.99, 0.99)
    pcm16_samples = (clipped * 32767.0).round().astype('<i2')

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16_samples.tobytes())

    duration = len(pcm16_samples) / sample_rate
    print(f'Saved: {path} | Duration: {duration:.2f}s | Samples: {len(pcm16_samples)}')


# ============================================================
# TTS INFERENCE ENGINE
# ============================================================

class PocketTTS:
    """Fast, CPU-optimized streaming Persian TTS engine based on Pocket TTS."""

    def __init__(
        self,
        config: PocketTTSConfig = PocketTTSConfig(),
        audio_validator: Callable[[str, np.ndarray], bool] | None = None,
    ) -> None:
        self.config = config
        self.audio_validator = audio_validator
        self.model_dir = Path(self.config.model_dir).resolve()
        self._verify_files()

        # Load and resolve paths in configuration dictionary
        config_path = self.model_dir / 'farsi.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)

        # _verify_files() already requires these local files to exist.
        cfg_dict['weights_path'] = str((self.model_dir / 'model.safetensors').resolve())
        lookup_table = cfg_dict.setdefault('flow_lm', {}).setdefault('lookup_table', {})
        lookup_table['tokenizer_path'] = str((self.model_dir / 'tokenizer.model').resolve())

        pydantic_config = Config(**cfg_dict)
        self.model = TTSModel._from_pydantic_config_with_weights(
            pydantic_config,
            temp=self.config.temperature,
            sampler_decode_steps=1,
            noise_clamp=None,
            eos_threshold=self.config.eos_threshold,
            origin=config_path,
        )

        self.tokenizer = self.model.flow_lm.conditioner.tokenizer.sp
        self.prompt_state = None
        self.set_voice(self.config.voice_path, max_sec=self.config.voice_prompt_max_sec)

    def _verify_files(self) -> None:
        """Ensure all required model binaries and configs exist."""
        required = ['farsi.yaml', 'model.safetensors', 'tokenizer.model']
        missing = [f for f in required if not (self.model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f'Missing required Pocket TTS files in {self.model_dir}: {", ".join(missing)}')

    def set_voice(self, voice_path: Path | str, max_sec: float | None = None) -> None:
        """Pre-encode reference voice audio prompt to cache speaker characteristics."""
        resolved = Path(voice_path).expanduser().resolve()
        if not resolved.exists():
            # Fallback to check inside model_dir
            if (self.model_dir / voice_path).exists():
                resolved = (self.model_dir / voice_path).resolve()
            else:
                raise FileNotFoundError(f'Voice prompt file not found: {voice_path}')

        voice_file = str(resolved)
        voice_duration_limit = max_sec if max_sec is not None else self.config.voice_prompt_max_sec

        if voice_duration_limit > 0:
            try:
                wav, sr = sphn.read(voice_file)
                keep_samples = int(voice_duration_limit * sr)
                if wav.shape[-1] > keep_samples:
                    temp_dir = Path(tempfile.mkdtemp())
                    trimmed_path = temp_dir / 'voice_prompt.wav'
                    sphn.write_wav(str(trimmed_path), wav.mean(axis=0)[:keep_samples].astype('float32'), int(sr))
                    voice_file = str(trimmed_path)
            except Exception:
                pass

        self.prompt_state = self.model.get_state_for_audio_prompt(voice_file)

    def synthesize_stream(self, text: str) -> Generator[np.ndarray, None, None]:
        """Yield synthesized audio arrays chunk by chunk for real-time streaming."""
        if not has_spoken_text(text):
            return
        normalized = normalize_text(text)
        if not has_spoken_text(normalized):
            return

        chunks = split_text(
            normalized,
            count_tokens=lambda s: len(self.tokenizer.encode(s)),
            max_tokens=self.config.max_tokens,
            keep_punct_boundaries=self.config.pause_at_punct,
            min_tokens=self.config.min_tokens,
        )
        chunks = [chunk for chunk in chunks if has_spoken_text(chunk)]
        if not chunks:
            return

        pause_samples = int(self.config.sample_rate * self.config.chunk_pause_seconds)
        pause = np.zeros(pause_samples, dtype=np.float32)

        join_samples = int(self.config.sample_rate * self.config.join_seconds)
        join = np.zeros(join_samples, dtype=np.float32)

        for index, chunk in enumerate(chunks):
            chunk_audio = self._generate_checked_audio(chunk)
            chunk_audio = trim_silence(chunk_audio, self.config.sample_rate)

            # Apply 10ms micro fade-out to avoid edge pop artifacts
            fade_len = min(240, len(chunk_audio))
            if fade_len > 0:
                chunk_audio[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

            yield chunk_audio

            if index < len(chunks) - 1:
                if chunk.rstrip().endswith(BREAK_CHARS):
                    yield pause
                else:
                    yield join

    def _generate_checked_audio(self, text: str) -> np.ndarray:
        """Retry mismatched speech before it reaches the playback queue."""
        for attempt in range(3):
            audio = self.model.generate_audio(
                self.prompt_state,
                text,
                frames_after_eos=self.config.frames_after_eos,
                copy_state=True,
            )
            waveform = np.asarray(
                audio.cpu().numpy() if hasattr(audio, 'cpu') else audio,
                dtype=np.float32,
            ).reshape(-1)
            if self.audio_validator is None or self.audio_validator(text, waveform):
                return waveform
            if attempt < 2:
                print('[*] TTS speech did not match the text; regenerating...')
        raise RuntimeError('TTS speech did not match the requested text after 3 attempts.')

    def synthesize(self, text: str) -> np.ndarray:
        """Batch generation returning a single concatenated waveform."""
        segments = list(self.synthesize_stream(text))
        if not segments:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(segments)


# ============================================================
# PLAYBACK & ENTRY POINT
# ============================================================

def stream_sentences(token_generator: Iterable[str]) -> Generator[str, None, None]:
    """Buffer tokens and yield complete sentences as punctuation marks arrive."""
    delimiters = {'.', '!', '؟', '?', '\n'}
    buffer = ''
    for token in token_generator:
        buffer += token
        start = 0
        for index, char in enumerate(buffer):
            if char not in delimiters:
                continue
            sentence = buffer[start:index + 1].strip()
            start = index + 1
            if has_spoken_text(sentence):
                yield sentence
        buffer = buffer[start:]

    if has_spoken_text(buffer):
        yield buffer.strip()


def play_sentence_stream(
    engine: PocketTTS,
    sentence_iterator: Iterable[str],
    on_playback_start: Callable[[], None] | None = None,
) -> None:
    """Prepare audio ahead on one worker while the caller plays queued chunks."""
    audio_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=10)
    cancelled = threading.Event()
    errors: list[Exception] = []

    def enqueue(chunk: np.ndarray | None) -> bool:
        while not cancelled.is_set():
            try:
                audio_queue.put(chunk, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def producer() -> None:
        try:
            for sentence in sentence_iterator:
                if cancelled.is_set():
                    return
                sent = sentence.strip()
                if not sent:
                    continue
                for chunk in engine.synthesize_stream(sent):
                    if cancelled.is_set():
                        return
                    if chunk.size and not enqueue(chunk):
                        return
        except Exception as exc:
            errors.append(exc)
        finally:
            enqueue(None)

    worker = threading.Thread(target=producer, daemon=True)
    sample_rate = engine.config.sample_rate
    started = False
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
        worker.start()
        try:
            while True:
                chunk = audio_queue.get()
                if chunk is None:
                    break
                if not started:
                    started = True
                    if on_playback_start:
                        on_playback_start()
                stream.write(chunk)
        finally:
            cancelled.set()
            worker.join()

    if errors:
        raise errors[0]


def play_buffered(
    engine: PocketTTS,
    text: str,
    on_playback_start: Callable[[], None] | None = None,
) -> None:
    """Stream audio chunks via background thread into sounddevice output."""
    play_sentence_stream(engine, [text], on_playback_start=on_playback_start)


def main() -> None:
    config = PocketTTSConfig()
    engine = PocketTTS(config)

    demo_text = (
        'سلام! مدل جدید پاکت تی‌تی‌اس با موفقیت راه‌اندازی شد. '
        'این مدل روی سی‌پی‌یو اجرا میشه و سرعت پردازش بسیار بالایی داره.'
        'امیدارم خوب باشی.'
        'بای بای. '
    )

    print('[*] Synthesizing demo sentence...')
    audio = engine.synthesize(demo_text)
    out_path = Path('./output_pocket.wav')
    save_wav(out_path, audio, engine.config.sample_rate)

    print('[*] Playing via sounddevice...')
    try:
        play_buffered(engine, demo_text)
    except Exception as e:
        print(f'Playback skipped or audio device unavailable: {e}')


if __name__ == '__main__':
    main()
