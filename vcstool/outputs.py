from dataclasses import dataclass
import re


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

    def fix_detached_head(self):
        """If The local branch is in a detached head state, parse the output to extract the hash
        (or tag)."""
        match = re.match(r"\(HEAD detached at (\S+)\)", self.local_branch)
        if match is not None:
            self.local_branch = "HEAD detached"
            self.local_hash = match[1]
