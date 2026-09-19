"""Reuse project 02's STT/TTS without importing its models.py or conversation DB."""
import importlib.util
import logging
import sys
from difflib import SequenceMatcher
from dataclasses import asdict

from config import AI_DIR


def load_ai_module(name):
    alias = f'_secretary_ai_{name}'
    if alias in sys.modules:
        return sys.modules[alias]
    path = AI_DIR / f'{name}.py'
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module  # Required by dataclasses during module execution.
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module


class Engines:
    def __init__(self, settings):
        from openai import OpenAI
        from audio import resample
        self.settings = settings
        self.client = OpenAI(api_key='ollama', base_url=settings.llm_url,
                             timeout=settings.request_timeout, max_retries=0)
        try:
            print('[AI] Checking Ollama...')
            self.client.chat.completions.create(model=settings.llm_model,
                                               messages=[{'role': 'user', 'content': 'سلام'}], max_tokens=1)
            stt = load_ai_module('stt')
            tts = load_ai_module('tts')
            print('[AI] Loading Shenava and Silero on CPU...')
            self.asr = stt.ShenavaASR(stt.ASRConfig(model_path=settings.asr_model,
                                                 min_rms_energy=settings.min_rms, use_cuda=False))

            def validate(expected, waveform):
                actual = self.asr.transcribe(resample(waveform, 24000, self.asr.config.sample_rate))
                def comparable(text):
                    return ''.join(c for c in tts.normalize_text(text) if c.isalnum())
                similarity = SequenceMatcher(None, comparable(expected), comparable(actual)).ratio()
                matches = bool(actual) and similarity >= 0.5
                if not matches:
                    logging.getLogger(__name__).warning(
                        'TTS verification failed: expected=%r recognized=%r similarity=%.2f',
                        expected, actual, similarity,
                    )
                return matches

            print('[AI] Loading Persian Pocket TTS on CPU...')
            self.tts = tts.PocketTTS(tts.TTSConfig(model_dir=settings.tts_dir,
                                                 voice_path=settings.voice,
                                                 voice_prompt_max_sec=settings.voice_prompt_seconds,
                                                 use_cuda=False),
                                     audio_validator=validate)
            self.prompt = settings.prompt_file.read_text(encoding='utf-8').strip()
            if not self.prompt:
                raise ValueError('Secretary prompt is empty')
            from speech_cache import file_identity, load_or_create
            greeting_key = {
                'version': 1,
                'text': settings.greeting,
                'tts_config': asdict(self.tts.config),
                'files': [file_identity(path) for path in (
                    settings.voice, settings.tts_dir / 'model.safetensors',
                    settings.tts_dir / 'tokenizer.model', settings.tts_dir / 'farsi.yaml',
                    AI_DIR / 'tts.py',
                )],
            }
            # Preload the saved greeting before answering any call: no TTS or disk I/O
            # is needed when the call connects. Other fixed messages stay in memory.
            print('[AI] Preparing greeting and silence messages...')
            greeting_audio = load_or_create(settings.greeting_file, greeting_key,
                                            self.sample_rate, lambda: self.tts.synthesize(settings.greeting))
            self.cached_audio = {text: self.tts.synthesize(text) for text in
                                 (settings.silence_message, settings.goodbye)}
            self.cached_audio[settings.greeting] = greeting_audio
        except BaseException:
            self.client.close()
            raise

    def reply(self, history):
        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[{'role': 'system', 'content': self.prompt}, *history],
            max_tokens=220, extra_body={'think': False, 'enable_thinking': False},
        )
        text = response.choices[0].message.content or ''
        # Do not speak a model's reasoning if it ignores the thinking option.
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.split('<think>', 1)[0].strip()
        if not text:
            raise RuntimeError('Ollama returned no spoken reply')
        return text

    def speech(self, text):
        if text in self.cached_audio:
            return iter([self.cached_audio[text]])
        return self.tts.synthesize_stream(text)

    @property
    def sample_rate(self):
        return self.tts.config.sample_rate

    def close(self):
        self.client.close()
