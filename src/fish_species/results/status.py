"""Pure filesystem-state inference, deliberately separate from scheduler state."""

from __future__ import annotations

from .schemas import FilesystemRunState


def infer_filesystem_state(
    *,
    raw_exit_status: str | None,
    failed_table_status: str | None,
    terminal_metrics_present: bool,
    updated_at: float,
    now: float,
    active_window_seconds: float,
) -> tuple[FilesystemRunState, str]:
    """Infer state from files only; this never claims a SLURM job is live."""

    if failed_table_status and failed_table_status != "0":
        return (
            FilesystemRunState.FAILED,
            f"failed_runs.csv status={failed_table_status}",
        )
    if raw_exit_status is not None and raw_exit_status != "0":
        return FilesystemRunState.FAILED, f"run_status.txt={raw_exit_status}"
    if raw_exit_status == "0" and terminal_metrics_present:
        return (
            FilesystemRunState.COMPLETED,
            "run_status.txt=0 and terminal metrics present",
        )
    if raw_exit_status == "0":
        return (
            FilesystemRunState.INCOMPLETE,
            "run_status.txt=0 but terminal metrics are absent",
        )
    if terminal_metrics_present:
        return (
            FilesystemRunState.COMPLETED,
            "terminal metrics present; no status marker",
        )
    if now - updated_at <= active_window_seconds:
        return (
            FilesystemRunState.POSSIBLY_ACTIVE,
            "recent partial artifacts; scheduler state not queried",
        )
    return (
        FilesystemRunState.INCOMPLETE,
        "partial artifacts and no terminal status",
    )
