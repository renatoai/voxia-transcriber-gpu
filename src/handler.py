"""
VoxiaHub Transcription — RunPod Serverless Handler
WhisperX (large-v3) + Pyannote diarization on GPU.
Output compatible with ElevenLabs Scribe API.
"""

import os
import time
import base64
import tempfile
import logging

import torch
import torchaudio
import whisperx
import requests as http_requests
from pyannote.audio import Pipeline as PyannotePipeline

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

diarize_pipeline = None
if HF_TOKEN:
    logger.info("Loading Pyannote diarization...")
    t0 = time.time()
    diarize_pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN,
    )
    diarize_pipeline.to(torch.device(DEVICE))
    logger.info(f"Pyannote loaded in {time.time() - t0:.1f}s")
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


def transcribe_audio(audio_path: str, language: str = "pt"):
    audio = whisperx.load_audio(audio_path)

    result = whisper_model.transcribe(
        audio,
        language=language,
        batch_size=WHISPER_BATCH_SIZE,
    )

    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )

    words = []
    full_text_parts = []

    for segment in result.get("segments", []):
        text = segment.get("text", "").strip()
        if text:
            full_text_parts.append(text)
        for w in segment.get("words", []):
            words.append({
                "text": " " + w.get("word", w.get("text", "")),
                "start": round(w.get("start", 0), 3),
                "end": round(w.get("end", 0), 3),
                "type": "word",
                "speaker_id": None,
            })
            words.append({
                "text": " ", "start": None, "end": None,
                "type": "spacing", "speaker_id": None,
            })

    if words and words[-1]["type"] == "spacing":
        words.pop()

    return " ".join(full_text_parts), words, language


def diarize_audio(audio_path: str):
    if diarize_pipeline is None:
        return None
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return diarize_pipeline({"waveform": waveform, "sample_rate": sample_rate})


def assign_speakers(words: list, diarize_result) -> list:
    if diarize_result is None:
        for w in words:
            if w["type"] == "word":
                w["speaker_id"] = "speaker_0"
        return words

    turns = []
    for turn, _, speaker in diarize_result.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, speaker))

    unique_speakers = []
    for _, _, spk in turns:
        if spk not in unique_speakers:
            unique_speakers.append(spk)

    for w in words:
        if w["type"] != "word" or w["start"] is None:
            continue
        word_mid = (w["start"] + w["end"]) / 2
        best_speaker = None
        best_overlap = 0
        for turn_start, turn_end, speaker in turns:
            overlap = max(0, min(w["end"], turn_end) - max(w["start"], turn_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        if best_speaker is None:
            min_dist = float("inf")
            for turn_start, turn_end, speaker in turns:
                dist = min(abs(word_mid - turn_start), abs(word_mid - turn_end))
                if dist < min_dist:
                    min_dist = dist
                    best_speaker = speaker
        if best_speaker:
            w["speaker_id"] = f"speaker_{unique_speakers.index(best_speaker)}"
        else:
            w["speaker_id"] = "speaker_0"

    for i, w in enumerate(words):
        if w["type"] == "spacing" and i > 0 and words[i - 1].get("speaker_id"):
            w["speaker_id"] = words[i - 1]["speaker_id"]

    return words


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

        # Transcribe
        logger.info(f"Transcribing: {filename}")
        t0 = time.time()
        full_text, words, lang = transcribe_audio(audio_path, language)
        logger.info(f"Transcription: {time.time() - t0:.1f}s — {len(words)} words")

        # Diarize
        if should_diarize and diarize_pipeline is not None:
            logger.info("Diarizing...")
            t0 = time.time()
            dr = diarize_audio(audio_path)
            logger.info(f"Diarization: {time.time() - t0:.1f}s")
            words = assign_speakers(words, dr)
        else:
            for w in words:
                if w["type"] == "word":
                    w["speaker_id"] = "speaker_0"

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
        logger.error(f"Failed: {e}")
        return {"error": str(e)}

    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


runpod.serverless.start({"handler": handler})
