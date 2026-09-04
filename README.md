# runpod_serverless_chatterbox_es_latam

RunPod serverless worker for voice cloning using **Chatterbox Multilingual**
(Resemble AI), specifically the **Latin American Spanish finetune**
(`ResembleAI/Chatterbox-Multilingual-es-mx-latam`) — MIT licensed, not gated.

## Why this exists (vs the F5-TTS worker)

`runpod_serverless_f5tts_spanish` (F5-TTS + `jpgallegoar/F5-Spanish`) works,
but real-world testing found it unreliable on **very short utterances**
(1-3 words render with garbled diction, e.g. "que" -> "kue") — a known class
of issue with duration-predictor-based TTS models when given too little text
to estimate timing from. F5-Spanish also has no dedicated Latin American
checkpoint (it's a broad multi-dialect mix, mostly Peninsular-leaning
training data).

Chatterbox is a different architecture (not duration-predictor based in the
same way) and has an official, MIT-licensed, ungated Latin American Spanish
finetune. Worth a real comparison.

## Loading the model — important correction

The model card's Python usage example references a `model_name=` kwarg on
`from_pretrained()`. **That kwarg does not exist** in the actual source
(`chatterbox/mtl_tts.py`) — `from_pretrained()` only ever downloads from the
hardcoded base repo `ResembleAI/chatterbox`. Confirmed by reading the raw
source directly rather than trusting the model card.

To actually use the es-mx-latam finetune, its files have to be combined with
one file from the base repo, into a single local directory, then loaded via
`from_local()`:

| File | Source repo | Notes |
|---|---|---|
| `ve.pt` | `ResembleAI/chatterbox` (base) | Voice encoder — generic/language-agnostic |
| `t3_es_mx_latam.safetensors` | `...-es-mx-latam` (finetune) | The actual Spanish-Latam finetune |
| `grapheme_mtl_merged_expanded_v1.json` | `...-es-mx-latam` (finetune) | Use the finetune's copy, not the base's |
| `s3gen_v3.pt` → renamed `s3gen.pt` | `...-es-mx-latam` (finetune) | `from_local()` hardcodes the filename `s3gen.pt`; the finetune repo ships it as `s3gen_v3.pt`, bundled as a matched pair with the t3 finetune |

`conds.pt` (base repo, optional fallback default voice) is intentionally
**not** downloaded — the handler always passes `audio_prompt_path`, so the
fallback conditioning is never used.

See `app/handler.py` (`assemble_checkpoint_dir()`) for the exact
implementation. Weights are baked into the Docker image at **build time**
(not downloaded on first request) to avoid multi-minute cold starts.

## Request format

```json
{
  "input": {
    "voice_url": "https://.../reference.wav",
    "text": "Text to synthesize in the cloned voice",
    "language_id": "es",
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8
  }
}
```

Unlike the F5-TTS worker, **no `ref_text` is needed** — Chatterbox's voice
cloning only needs the reference audio itself, not a transcript of it.

## Response format

```json
{ "output_audio_base64": "...", "format": "wav" }
```

Base64 inline, same reasoning as the F5-TTS worker: RunPod's S3-compatible
API supports neither ACLs nor pre-signed URLs.

## Model stays warm

The model loads **once** at container startup (`if __name__ == "__main__"`
block, before `runpod.serverless.start`) and is reused across every job a
warm worker handles. The F5-TTS worker instead shelled out to a fresh CLI
process per request, reloading the full model from disk every single time —
this should be meaningfully faster per-request once a worker is warm.

## Watermarking

Chatterbox embeds an imperceptible watermark (`perth.PerthImplicitWatermarker`)
in all generated audio by default (Resemble AI's responsible-AI feature).
This is not configurable via the public API used here.
