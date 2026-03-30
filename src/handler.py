"""
VoxiaHub Transcription — RunPod Serverless Handler
WhisperX (large-v3) + WhisperX DiarizationPipeline on GPU.
Output compatible with ElevenLabs Scribe API.
"""

import os
import time
import base64
import tempfile
import logging
import gc

import torch
import whisperx
import requests as http_requests

import runpod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voxia-gpu")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "16"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Load models at cold start (stays in memory between requests)
# ---------------------------------------------------------------------------
logger.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

logger.info(f"Loading WhisperX: {WHISPER_MODEL_SIZE} ({WHISPER_COMPUTE_TYPE})")
t0 = time.time()
whisper_model = whisperx.load_model(
    WHISPER_MODEL_SIZE,
    device=DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
    language="pt",
)
logger.info(f"WhisperX loaded in {time.time() - t0:.1f}s")

logger.info("Loading alignment model (pt)...")
t0 = time.time()
align_model, align_metadata = whisperx.load_align_model(
    language_code="pt",
    device=DEVICE,
)
logger.info(f"Alignment loaded in {time.time() - t0:.1f}s")

# Use WhisperX's own DiarizationPipeline wrapper (not raw pyannote)
diarize_model = None
if HF_TOKEN:
    logger.info("Loading WhisperX DiarizationPipeline...")
    t0 = time.time()
    from whisperx.diarize import DiarizationPipeline
    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        token=HF_TOKEN,
        device=DEVICE,
    )
    logger.info(f"Diarization pipeline loaded in {time.time() - t0:.1f}s")
else:
    logger.warning("HF_TOKEN not set — diarization disabled.")

logger.info("Ready.")


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
def download_audio(url: str) -> str:
    resp = http_requests.get(url, timeout=120)
    resp.raise_for_status()
    suffix = ".mp3"
    for ext in [".wav", ".ogg", ".m4a", ".flac", ".opus"]:
        if ext in url.lower():
            suffix = ext
            break
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(resp.content)
    tmp.close()
    return tmp.name


def transcribe_and_diarize(audio_path: str, language: str = "pt",
                           should_diarize: bool = True,
                           min_speakers: int = None,
                           max_speakers: int = None):
    """Transcribe + align + diarize using WhisperX native pipeline."""

    audio = whisperx.load_audio(audio_path)

    # 1. Transcribe
    result = whisper_model.transcribe(
        audio,
        language=language,
        batch_size=WHISPER_BATCH_SIZE,
    )

    # 2. Align
    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )

    # 3. Diarize using WhisperX DiarizationPipeline (returns proper DataFrame)
    if should_diarize and diarize_model is not None:
        logger.info(f"Running diarization (min_speakers={min_speakers}, max_speakers={max_speakers})...")
        t0 = time.time()

        diarize_segments = diarize_model(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        result = whisperx.assign_word_speakers(diarize_segments, result)
        logger.info(f"Diarization done in {time.time() - t0:.1f}s")

        # Free GPU memory
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Convert to ElevenLabs Scribe format
    words = []
    full_text_parts = []

    # Map speaker labels to speaker_0, speaker_1, etc.
    speaker_map = {}
    speaker_idx = 0

    for segment in result.get("segments", []):
        text = segment.get("text", "").strip()
        if text:
            full_text_parts.append(text)

        seg_speaker = segment.get("speaker", "SPEAKER_00")

        for w in segment.get("words", []):
            w_speaker = w.get("speaker", seg_speaker)
            if w_speaker not in speaker_map:
                speaker_map[w_speaker] = f"speaker_{speaker_idx}"
                speaker_idx += 1

            words.append({
                "text": " " + w.get("word", w.get("text", "")),
                "start": round(w.get("start", 0), 3) if w.get("start") is not None else None,
                "end": round(w.get("end", 0), 3) if w.get("end") is not None else None,
                "type": "word",
                "speaker_id": speaker_map.get(w_speaker, "speaker_0"),
            })
            words.append({
                "text": " ", "start": None, "end": None,
                "type": "spacing", "speaker_id": speaker_map.get(w_speaker, "speaker_0"),
            })

    if words and words[-1]["type"] == "spacing":
        words.pop()

    logger.info(f"Speakers found: {speaker_map}")

    return " ".join(full_text_parts), words, language


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def handler(event):
    """
    Input:
        audio_url: str       — URL to download audio
        audio_base64: str    — OR base64-encoded audio
        language: str        — default "pt"
        diarize: bool        — default true
        min_speakers: int    — optional, hint for diarization
        max_speakers: int    — optional, hint for diarization
        filename: str        — optional, for logging

    Output (ElevenLabs Scribe compatible):
        text: str
        words: [{text, start, end, type, speaker_id}, ...]
        language_code: str
        language_probability: float
    """
    input_data = event.get("input", {})

    audio_url = input_data.get("audio_url")
    audio_b64 = input_data.get("audio_base64")
    language = input_data.get("language", "pt")
    should_diarize = input_data.get("diarize", True)
    min_speakers = input_data.get("min_speakers", 2)
    max_speakers = input_data.get("max_speakers", None)
    filename = input_data.get("filename", "audio.mp3")

    if not audio_url and not audio_b64:
        return {"error": "audio_url or audio_base64 required"}

    t_total = time.time()
    audio_path = None

    try:
        # Get audio
        if audio_url:
            logger.info(f"Downloading: {filename}")
            audio_path = download_audio(audio_url)
        else:
            logger.info(f"Decoding base64: {filename}")
            suffix = os.path.splitext(filename)[1] or ".mp3"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(base64.b64decode(audio_b64))
            tmp.close()
            audio_path = tmp.name

        # Transcribe + diarize
        logger.info(f"Processing: {filename}")
        full_text, words, lang = transcribe_and_diarize(
            audio_path, language, should_diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        elapsed = time.time() - t_total
        logger.info(f"Total: {elapsed:.1f}s for {filename}")

        return {
            "text": full_text,
            "words": words,
            "language_code": lang,
            "language_probability": 0.99,
            "processing_time": round(elapsed, 1),
        }

    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return {"error": str(e)}

    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


runpod.serverless.start({"handler": handler})

