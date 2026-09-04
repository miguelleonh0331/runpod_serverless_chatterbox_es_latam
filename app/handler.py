"""
RunPod serverless handler for Chatterbox Multilingual TTS, using the
Latin American Spanish finetune (ResembleAI/Chatterbox-Multilingual-es-mx-latam)
instead of F5-TTS (jpgallegoar/F5-Spanish), which showed unreliable diction on
very short utterances (root-caused during the runpod_serverless_f5tts_spanish
project) and has no dedicated Latin American Spanish checkpoint of its own.

Assembly note (reverse-engineered from chatterbox's mtl_tts.py source, since
the model card's usage example references a `model_name=` kwarg that does not
actually exist in from_pretrained()):

  ChatterboxMultilingualTTS.from_local(ckpt_dir, device, t3_model=...) expects
  exactly these files in ckpt_dir: ve.pt, s3gen.pt,
  grapheme_mtl_merged_expanded_v1.json, and the t3 safetensors file.

  - ve.pt (voice encoder) is generic/language-agnostic -> pulled from the base
    repo ResembleAI/chatterbox.
  - t3_es_mx_latam.safetensors, grapheme_mtl_merged_expanded_v1.json, and
    s3gen_v3.pt (renamed locally to s3gen.pt) all come from the es-mx-latam
    finetune repo -- they're bundled together there as a matched set, so they
    are used together rather than mixed with the base repo's plain s3gen.pt.

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
T3_MODEL_FILENAME = "t3_es_mx_latam.safetensors"

CKPT_DIR = Path("/opt/chatterbox_ckpt")
WORK_ROOT = Path("/tmp/chatterbox_jobs")

DEFAULT_LANGUAGE_ID = "es"

_model = None  # loaded once, reused across jobs on a warm worker


def assemble_checkpoint_dir() -> Path:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    ve_path = hf_hub_download(repo_id=BASE_REPO_ID, filename="ve.pt")
    t3_path = hf_hub_download(repo_id=FINETUNE_REPO_ID, filename=T3_MODEL_FILENAME)
    grapheme_path = hf_hub_download(repo_id=FINETUNE_REPO_ID, filename="grapheme_mtl_merged_expanded_v1.json")
    s3gen_path = hf_hub_download(repo_id=FINETUNE_REPO_ID, filename="s3gen_v3.pt")

    def link(src: str, name: str):
        dst = CKPT_DIR / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

    link(ve_path, "ve.pt")
    link(t3_path, T3_MODEL_FILENAME)
    link(grapheme_path, "grapheme_mtl_merged_expanded_v1.json")
    link(s3gen_path, "s3gen.pt")  # renamed: from_local() expects exactly "s3gen.pt"

    return CKPT_DIR


def load_model():
    global _model
    if _model is not None:
        return _model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[chatterbox] loading model on {device}...")
    ckpt_dir = assemble_checkpoint_dir()
    _model = ChatterboxMultilingualTTS.from_local(ckpt_dir, device, t3_model=T3_MODEL_FILENAME)
    print("[chatterbox] model loaded and warm.")
    return _model


def download_file(url: str, local_path: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


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
