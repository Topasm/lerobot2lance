"""Small Hugging Face upload helper for converted Lance bundles."""

from __future__ import annotations

from pathlib import Path


def upload_lance_bundle_to_hub(
    bundle_root: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    revision: str | None = None,
    commit_message: str | None = None,
    tag: str | None = None,
    tag_message: str | None = None,
) -> str | None:
    """Upload ``bundle_root`` to a Hugging Face dataset repo.

    Returns the commit URL when the installed ``huggingface_hub`` version
    exposes one.
    """

    bundle_root = Path(bundle_root)
    if not bundle_root.exists():
        raise FileNotFoundError(f"Bundle root not found: {bundle_root}")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "HF upload requires huggingface_hub. Install with "
            '`pip install "lerobot2lance[hub]"` or `pip install huggingface_hub`.'
        ) from exc

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    upload_kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "folder_path": str(bundle_root),
        "path_in_repo": ".",
        "commit_message": commit_message
        or f"Upload Lance dataset {bundle_root.name}",
    }
    if revision:
        upload_kwargs["revision"] = revision
    result = api.upload_folder(**upload_kwargs)
    commit_url = getattr(result, "commit_url", None)

    if tag:
        kwargs = {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "tag": tag,
            "tag_message": tag_message or f"Release {tag}",
        }
        if revision:
            kwargs["revision"] = revision
        try:
            api.create_tag(**kwargs, exist_ok=False)
        except TypeError:
            api.create_tag(**kwargs)

    return commit_url
