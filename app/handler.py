"""
RunPod serverless handler for Chatterbox Multilingual TTS, using the
Latin American Spanish finetune (ResembleAI/Chatterbox-Multilingual-es-mx-latam)
instead of F5-TTS (jpgallegoar/F5-Spanish), which showed unreliable diction on
very short utterances (root-caused during the runpod_serverless_f5tts_spanish
project) and has no dedicated Latin American Spanish checkpoint of its own.

Assembly note (reverse-engineered from the ACTUAL installed package source,
not just the model card or the GitHub repo's HEAD -- both were misleading):

  - The model card's usage example references a `model_name=` kwarg on
    from_pretrained(). That kwarg does not exist anywhere.
  - GitHub `master` has a `from_local(ckpt_dir, device, t3_model=...)` with an
    overridable T3 filename -- but that's AHEAD of what's actually installed.
  - The real installed package (chatterbox-tts==0.1.7, verified by
    downloading the wheel and reading it directly) has
    `from_local(ckpt_dir, device)` with NO override parameter at all: it
    hardcodes the T3 filename to load as exactly "t3_mtl23ls_v2.safetensors".

  So the finetune's T3 file has to be placed locally UNDER THAT HARDCODED
  NAME (not its own name) for from_local() to find it. Files placed in
  ckpt_dir:
  - ve.pt <- base repo ResembleAI/chatterbox (voice encoder, generic/
    language-agnostic).
  - t3_mtl23ls_v2.safetensors <- the finetune's t3_es_mx_latam.safetensors,
    renamed to the hardcoded name from_local() expects.
  - grapheme_mtl_merged_expanded_v1.json <- the finetune repo's copy (not
    the base repo's) -- this is the T3 tokenizer's vocab, paired with the T3
    finetune, unrelated to s3gen.
  - s3gen.pt <- the BASE repo's plain s3gen.pt, NOT the finetune's
    s3gen_v3.pt. Tried the finetune's s3gen_v3.pt first (it's bundled
    alongside t3_es_mx_latam.safetensors in that repo, looked like an
    intentional matched pair) and it failed at load time with
    `RuntimeError: Missing key(s) in state_dict: "tokenizer._mel_filters",
    "tokenizer.window"` -- s3gen_v3.pt is built for a newer S3Gen
    architecture (with an internal tokenizer submodule) than what's actually
    implemented in the installed chatterbox-tts==0.1.7. The base repo's
    plain s3gen.pt is the one that architecturally matches this installed
    version (it's literally what from_pretrained()'s own allow_patterns
    downloads by default) -- s3gen is a generic vocoder-ish component, not
    inherently language/dialect-specific, so pairing the base s3gen.pt with
    the Spanish-Latam T3 finetune is expected to work correctly.

  conds.pt (base repo, optional fallback default voice) is intentionally NOT
  downloaded: generate() is always called here with audio_prompt_path set, so
  the fallback conds is never used.

The model is loaded ONCE at worker startup and kept warm in GPU memory across
requests (unlike the F5-TTS worker, which re-shelled to a fresh CLI process --
and reloaded the model from scratch -- on every single job).

Input (job['input']):
  voice_url  (required) - public URL to a reference voice clip (no exact
                           transcript needed -- unlike F5-TTS, Chatterbox does
                           not require ref_text).
  text       (required) - text to synthesize in the cloned voice.
  language_id (optional) - default "es".
  exaggeration, cfg_weight, temperature (optional) - passed through to
                           model.generate(), defaults match the library's own.

Output:
  { "output_audio_base64": "<base64 wav>", "format": "wav" }

Audio is returned as base64 directly in the job response, not uploaded to S3:
RunPod's S3-compatible API supports neither ACLs nor pre-signed URLs, so an
uploaded object can never be made publicly downloadable (see the `runpod`
project doc for the full investigation from the F5-TTS worker).
"""
import base64
import os
import uuid
from pathlib import Path

import requests
import runpod
import torch
import torchaudio as ta
from huggingface_hub import hf_hub_download
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

BASE_REPO_ID = "ResembleAI/chatterbox"
FINETUNE_REPO_ID = "ResembleAI/Chatterbox-Multilingual-es-mx-latam"
FINETUNE_T3_FILENAME = "t3_es_mx_latam.safetensors"
# The INSTALLED pip package (chatterbox-tts==0.1.7) hardcodes this exact
# filename inside from_local() -- it takes no parameter to override it
# (that parameter only exists on the GitHub `master` branch, which is ahead
# of the 0.1.7 release; confirmed by downloading and inspecting the actual
# wheel, not just trusting the repo's HEAD). So the finetune's T3 file has
# to be placed locally UNDER this name for from_local() to pick it up.
HARDCODED_T3_TARGET_NAME = "t3_mtl23ls_v2.safetensors"

CKPT_DIR = Path("/opt/chatterbox_ckpt")
WORK_ROOT = Path("/tmp/chatterbox_jobs")

DEFAULT_LANGUAGE_ID = "es"

_model = None  # loaded once, reused across jobs on a warm worker


def assemble_checkpoint_dir() -> Path:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    ve_path = hf_hub_download(repo_id=BASE_REPO_ID, filename="ve.pt")
    s3gen_path = hf_hub_download(repo_id=BASE_REPO_ID, filename="s3gen.pt")
    t3_path = hf_hub_download(repo_id=FINETUNE_REPO_ID, filename=FINETUNE_T3_FILENAME)
    grapheme_path = hf_hub_download(repo_id=FINETUNE_REPO_ID, filename="grapheme_mtl_merged_expanded_v1.json")

    def link(src: str, name: str):
        dst = CKPT_DIR / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

    link(ve_path, "ve.pt")
    link(s3gen_path, "s3gen.pt")
    link(t3_path, HARDCODED_T3_TARGET_NAME)  # renamed: from_local() hardcodes this filename
    link(grapheme_path, "grapheme_mtl_merged_expanded_v1.json")

    return CKPT_DIR


def load_model():
    global _model
    if _model is not None:
        return _model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[chatterbox] loading model on {device}...")
    ckpt_dir = assemble_checkpoint_dir()
    _model = ChatterboxMultilingualTTS.from_local(ckpt_dir, device)
    print("[chatterbox] model loaded and warm.")
    return _model


def download_file(url: str, local_path: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


# S3 tokenizer emits speech tokens at this rate (chatterbox/models/
# s3tokenizer/s3tokenizer.py). Confirmed via Deepgram STT on real output:
# generated audio consistently had one extra hallucinated word/sound
# ("Bueno.") tacked on at the very end that was not in the input text.
# Later chatterbox-tts versions (unreleased past 0.1.7, seen on GitHub
# master) fix this at the source by dropping the final speech token's
# audio before decoding, with the comment: "it is emitted just before EOS
# with degraded attention and decodes to ~40 ms of noise." The installed
# 0.1.7 does NOT have that fix, so it's replicated here as a post-process
# trim instead (same duration: one token's worth of samples at the
# model's output sample rate).
S3_TOKEN_RATE = 25


def trim_trailing_eos_noise(wav, sample_rate: int):
    trim_samples = max(1, round(sample_rate / S3_TOKEN_RATE))
    if wav.shape[-1] > trim_samples:
        return wav[..., :-trim_samples]
    return wav


def handler(job):
    job_input = job.get("input", {})

    voice_url = job_input.get("voice_url")
    text = job_input.get("text")
    language_id = job_input.get("language_id", DEFAULT_LANGUAGE_ID)
    exaggeration = float(job_input.get("exaggeration", 0.5))
    cfg_weight = float(job_input.get("cfg_weight", 0.5))
    temperature = float(job_input.get("temperature", 0.8))

    if not voice_url:
        return {"error": "voice_url is required"}
    if not text:
        return {"error": "text is required"}

    job_dir = WORK_ROOT / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = load_model()

        ref_audio_path = job_dir / "reference.wav"
        download_file(voice_url, ref_audio_path)

        wav = model.generate(
            text,
            language_id=language_id,
            audio_prompt_path=str(ref_audio_path),
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )

        wav = trim_trailing_eos_noise(wav, model.sr)

        output_path = job_dir / "output.wav"
        ta.save(str(output_path), wav, model.sr)

        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {"output_audio_base64": audio_base64, "format": "wav"}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    load_model()  # warm the model at container startup, not on first request
    runpod.serverless.start({"handler": handler})
