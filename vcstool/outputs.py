from dataclasses import dataclass
import re


@dataclass
class CompareOutput:
    """Simple dataclass which contains the output of the CompareCommand."""

    local_version: str
    remote_version: str
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
        """If The local version is in a detached head state, parse the output to extract the hash
        (or tag)."""
        match = re.match(r"\(HEAD detached at (\S+)\)", self.local_version)
        if match is not None:
            self.local_version = "HEAD detached"
            self.local_hash = match[1]
