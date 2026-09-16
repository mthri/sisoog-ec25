from __future__ import annotations

from pathlib import Path
from difflib import SequenceMatcher
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from llm import ChatSession, warmup_llm
from stt import ASRConfig, ShenavaASR
from tts import PocketTTS, TTSConfig, normalize_text, play_sentence_stream, stream_sentences


class VoiceAssistant:
    """Coordinates voice activity detection, speech recognition, LLM inference, and TTS playback."""

    def __init__(self) -> None:

        print('[*] Initializing ShenavaASR engine (CPU)...')
        self.asr = ShenavaASR(ASRConfig(
            model_path=Path(__file__).resolve().parent / 'shenava-koochik' / 'shenava-koochik-1.0.nemo',
            use_cuda=False,
        ))

        print('[*] Initializing PocketTTS engine (CPU)...')
        self.tts = PocketTTS(TTSConfig(use_cuda=False), audio_validator=self._validate_speech)

        print('[*] Pre-warming Ollama LLM engine...')
        warmup_llm(strict=True)
        self.chat = ChatSession()

    def _validate_speech(self, expected_text: str, audio: np.ndarray) -> bool:
        """Use the loaded ASR to catch reference speech or unrelated TTS output."""
        source_rate = self.tts.config.sample_rate
        target_rate = self.asr.config.sample_rate
        factor = gcd(source_rate, target_rate)
        audio = resample_poly(audio, target_rate // factor, source_rate // factor)
        recognized = self.asr.transcribe(audio)

        def comparable(text: str) -> str:
            # Ignore punctuation, spaces and Persian/Arabic spelling variants.
            return ''.join(char for char in normalize_text(text) if char.isalnum())

        expected = comparable(expected_text)
        actual = comparable(recognized)
        # Allow recognition mistakes, but reject speech unrelated to the request.
        matches = bool(actual) and SequenceMatcher(None, expected, actual).ratio() >= 0.5
        if not matches:
            print(f'[-] TTS verification: expected {expected_text!r}, recognized {recognized!r}')
        return matches

    def respond(self, text: str) -> None:
        """Request a reply, then generate and play audio incrementally."""
        print(f'You: {text}')
        print('[*] Generating response (microphone off)...')
        reply = self.chat.ask(text).strip()
        if not reply:
            print('[*] LLM returned an empty response.')
            return

        print(f'Assistant: {reply}')
        print('[*] Streaming speech (microphone off)...')
        play_sentence_stream(self.tts, stream_sentences([reply]))

    def run(self) -> None:
        """Listen again only after the preceding reply has finished playing."""
        print('[*] All models ready. Press Ctrl+C to stop.')
        while True:
            print('[*] Listening... Speak into the microphone.')
            # This closes the input stream before transcription returns.
            text = self.asr.listen_utterance(timeout=None).strip()
            if not text:
                continue
            try:
                self.respond(text)
            except Exception as exc:
                print(f'[-] Could not complete response: {exc}')


def main() -> None:
    try:
        VoiceAssistant().run()
    except KeyboardInterrupt:
        print('\n[*] Voice assistant stopped.')
    except Exception as exc:
        raise SystemExit(f'[-] Voice assistant failed: {exc}') from exc


if __name__ == '__main__':
    main()
