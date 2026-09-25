"""Checkpoint selection shared by training and the viewer."""

from pathlib import Path


def resolve_checkpoint_path(logdir: str, run_name: str | None):
    if not run_name:
        return None

    candidate = Path(run_name)
    if candidate.is_file():
        return str(candidate)

    if candidate.is_absolute():
        run_dir = candidate
    else:
        run_dir = Path(logdir) / run_name

    final_checkpoint = run_dir / "final" / "model_checkpoint.pth"
    if final_checkpoint.exists():
        return str(final_checkpoint)

    epoch_dirs = sorted(
        [
            path
            for path in run_dir.iterdir()
            if path.is_dir() and path.name.startswith("epoch_")
        ],
        key=lambda path: int(path.name.split("_")[-1]),
        reverse=True,
    )

    for epoch_dir in epoch_dirs:
        checkpoint_path = epoch_dir / "model_checkpoint.pth"
        if checkpoint_path.exists():
            return str(checkpoint_path)

    root_checkpoint = run_dir / "model_checkpoint.pth"
    if root_checkpoint.exists():
        return str(root_checkpoint)

    raise FileNotFoundError(
        f"Could not find a checkpoint for run '{run_name}' under {run_dir}"
    )
