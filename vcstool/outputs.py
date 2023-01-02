from dataclasses import dataclass


@dataclass
class CompareOutput:
    """Simple dataclass which contains the output of the CompareCommand."""

    local_branch: str
    remote_branch: str
    tag: str
    local_hash: str
    remote_hash: str
    remote: str
    ahead: int
    behind: int
    unstaged_changes: bool
    staged_changes: bool
    untracked_files: bool
    stashes: bool
