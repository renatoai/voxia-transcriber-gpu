# Voxia Transcriber GPU — RunPod Serverless

WhisperX (large-v3) + Pyannote diarization on GPU.
Drop-in replacement for ElevenLabs Scribe API.

## Deploy on RunPod Serverless

1. Connect this repo to RunPod Serverless
2. Set environment variables:
   - `HF_TOKEN` — HuggingFace token (required for diarization)
3. Deploy

## API

### Input
```json
{
  "input": {
    "audio_url": "https://example.com/audio.mp3",
    "language": "pt",
    "diarize": true
  }
}
```

Or with base64:
```json
{
  "input": {
    "audio_base64": "...",
    "filename": "audio.mp3",
    "language": "pt",
    "diarize": true
  }
}
```

### Output (ElevenLabs Scribe compatible)
```json
{
  "text": "Olá, como posso ajudar?",
  "words": [
    {"text": " Olá,", "start": 0.25, "end": 0.75, "type": "word", "speaker_id": "speaker_0"},
    {"text": " ", "start": null, "end": null, "type": "spacing", "speaker_id": "speaker_0"}
  ],
  "language_code": "pt",
  "language_probability": 0.99,
  "processing_time": 12.3
}
```
