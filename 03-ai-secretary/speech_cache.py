"""Persist verified greeting audio and invalidate it when its inputs change."""
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def load_or_create(path, key, sample_rate, synthesize):
    path = Path(path)
    metadata_path = path.with_suffix('.json')
    fingerprint = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if (metadata['key'] == fingerprint
                and metadata['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()):
            waveform, rate = sf.read(path, dtype='float32')
            if rate == sample_rate and waveform.ndim == 1 and waveform.size and np.isfinite(waveform).all():
                logging.getLogger(__name__).info('Loaded greeting from %s', path)
                return waveform
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        pass

    # Synthesis includes ASR validation; a failed generation never replaces the cache.
    waveform = np.asarray(synthesize(), dtype=np.float32).reshape(-1)
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError('Greeting synthesis returned empty or invalid audio')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = []
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.wav', delete=False) as f:
            wav_temp = Path(f.name)
            temporary.append(wav_temp)
        sf.write(wav_temp, waveform, sample_rate, subtype='PCM_16')
        metadata = {'key': fingerprint, 'sha256': hashlib.sha256(wav_temp.read_bytes()).hexdigest()}
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.json',
                                         encoding='utf-8', delete=False) as f:
            json_temp = Path(f.name)
            temporary.append(json_temp)
            json.dump(metadata, f)
        os.replace(wav_temp, path)
        os.replace(json_temp, metadata_path)
    finally:
        for temp in temporary:
            temp.unlink(missing_ok=True)
    logging.getLogger(__name__).info('Saved greeting to %s', path)
    # First run and later runs use exactly the same saved PCM samples.
    return sf.read(path, dtype='float32')[0]
