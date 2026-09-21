"""Configuration paths are relative to this project, never the working directory."""
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
AI_DIR = PROJECT_DIR.parent / '02-ai'


@dataclass(frozen=True)
class Settings:
    port: str = '/dev/ttyUSB0'  # Same default as 01-call-and-sms/main_v2.py.
    baudrate: int = 115200
    input_device: int | None = None
    output_device: int | None = None
    database: Path = PROJECT_DIR / 'secretary.db'
    prompt_file: Path = PROJECT_DIR / 'secretary_prompt.txt'
    asr_model: Path = AI_DIR / 'shenava-koochik/shenava-koochik-1.0.nemo'
    tts_dir: Path = AI_DIR / 'pocket-tts'
    voice: Path = AI_DIR / 'pocket-tts/example_voice.wav'
    voice_prompt_seconds: float = 2.0
    greeting_file: Path = PROJECT_DIR / 'cache/greeting.wav'
    llm_url: str = 'http://localhost:11434/v1'
    llm_model: str = 'gemma4:e2b'
    request_timeout: float = 60.0
    answer_delay: float = 2.0
    greeting_delay: float = 1.0
    thinking_enabled: bool = True  # Double beep while LLM/TTS prepares a reply.
    thinking_volume: float = 0.08  # PCM amplitude, from 0 (silent) to 1.
    ring_timeout: float = 20.0
    silence_timeout: float = 15.0
    max_silent_turns: int = 2
    max_call_seconds: float = 600.0
    phrase_limit: float = 20.0
    min_rms: float = 0.008
    greeting: str = 'سلام! من منشی هوش مصنوعی هستم. چه کمکی از دست من بر میاد؟'
    silence_message: str = 'صداتون رو نشنیدم. لطفاً اسمتون و پیامتون رو بگید.'
    goodbye: str = 'ممنون از تماستون. خداحافظ.'
