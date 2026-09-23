"""Download and verify the pinned base model used by this experiment."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import snapshot_download


from config import MODEL_ID, REVISION, MODEL_DIR

MODEL_FILES = (
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def sha256(path: Path) -> str:
    """Calculate a file hash without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Download the pinned snapshot, verify it, and record local metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID}")
    print(f"Pinned revision: {REVISION}")

    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=MODEL_DIR,
        allow_patterns=list(MODEL_FILES),
        token=False,
    )

    missing = [name for name in MODEL_FILES if not (MODEL_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")

    files = []
    for name in MODEL_FILES:
        path = MODEL_DIR / name
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})

    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "source": f"https://huggingface.co/{MODEL_ID}",
        "huggingface_hub_version": version("huggingface_hub"),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    manifest_path = MODEL_DIR / "download_manifest.json"
    if manifest_path.exists():
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verified_fields = ("model_id", "revision", "total_bytes", "files")
        mismatches = [
            field
            for field in verified_fields
            if saved_manifest.get(field) != manifest[field]
        ]
        if mismatches:
            raise ValueError(f"Model verification failed for: {', '.join(mismatches)}")
        print("Existing download manifest matches the local files")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("Created download manifest")

    print(f"Verified {len(files)} files ({manifest['total_bytes'] / 1_000_000:.1f} MB)")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
