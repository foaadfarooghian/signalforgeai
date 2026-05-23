"""Release evidence orchestration for SignalForge AI."""

from signalforgeai.release.candidate import (
    RELEASE_CANDIDATE_VERSION,
    run_release_candidate_check,
    write_release_training_evidence,
)

__all__ = [
    "RELEASE_CANDIDATE_VERSION",
    "run_release_candidate_check",
    "write_release_training_evidence",
]
