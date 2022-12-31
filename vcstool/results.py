from dataclasses import dataclass


@dataclass
class CompareResults:
    """Simple dataclass which contains the results of the CompareCommand."""

    local_branch: str
    remote_branch: str
    tag: str
    hash: str
    remote: str
    ahead: int
    behind: int
    unstaged_changes: bool
    staged_changes: bool
    untracked_files: bool
    stashes: bool
