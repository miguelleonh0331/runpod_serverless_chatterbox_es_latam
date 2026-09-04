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

## Loading the model — two corrections, found the hard way

Two different sources turned out to be misleading, in two different ways —
both only caught by reading actual source code instead of trusting docs:

1. The **model card**'s Python usage example references a `model_name=` kwarg
   on `from_pretrained()`. That kwarg does not exist anywhere.
2. **GitHub `master`** has a `from_local(ckpt_dir, device, t3_model=...)`
   with an overridable T3 filename — but that's ahead of what's actually
   published to PyPI. The **real installed package**
   (`chatterbox-tts==0.1.7`, confirmed by downloading the wheel directly and
   reading it) has `from_local(ckpt_dir, device)` with **no override
   parameter at all** — it hardcodes the T3 filename to load as exactly
   `t3_mtl23ls_v2.safetensors`.

First deploy failed with `TypeError: from_local() got an unexpected keyword
argument 't3_model'` — that's this exact mismatch (see commit history).

So to use the es-mx-latam finetune, its files have to be combined with one
file from the base repo into a single local directory, with the T3 file
placed under the **hardcoded name** `from_local()` expects, not its own:

| Local filename in `ckpt_dir` | Source repo | Real filename there | Notes |
|---|---|---|---|
| `ve.pt` | `ResembleAI/chatterbox` (base) | `ve.pt` | Voice encoder — generic/language-agnostic |
| `t3_mtl23ls_v2.safetensors` | `...-es-mx-latam` (finetune) | `t3_es_mx_latam.safetensors` | **Renamed** — `from_local()` hardcodes this exact filename regardless of which checkpoint you actually want loaded |
| `grapheme_mtl_merged_expanded_v1.json` | `...-es-mx-latam` (finetune) | same name | Use the finetune's copy, not the base's |
| `s3gen.pt` | `...-es-mx-latam` (finetune) | `s3gen_v3.pt` | **Renamed** — `from_local()` hardcodes `s3gen.pt`; the finetune ships it as `s3gen_v3.pt`, bundled as a matched pair with its own t3 |

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
