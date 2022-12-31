import argparse
import sys
import os
import abc
from typing import Any, Dict, List, Optional
from enum import IntEnum, IntFlag, auto

import prettytable as pt

from vcstool.crawler import find_repositories
from vcstool.executor import ansi, execute_jobs, generate_jobs
from vcstool.commands.import_ import get_repositories
from vcstool.streams import set_streams
from vcstool.outputs import CompareOutput

from .command import add_common_arguments
from .command import Command


class CompareCommand(Command):

    command = "compare"
    help = "Compare working copy to the repository list file"

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.progress = args.progress
        self.workers = args.workers
        self.debug = args.debug

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=f"{cls.help}", prog="vcs compare")
        group = parser.add_argument_group('"compare" command parameters')
        group.add_argument(
            "-i",
            "--input",
            type=str,
            default="workspace.repos",
            help="Path to the repository list file for the workspace.",
        )
        group.add_argument(
            "-p",
            "--progress",
            action="store_true",
            default=False,
            help="Show progress of the jobs during execution.",
        )
        group.add_argument(
            "-n",
            "--nested",
            action="store_true",
            default=False,
            help="Search the workspace for nested repositories.",
        )
        group.add_argument(
            "-s",
            "--significant",
            action="store_true",
            default=False,
            help="Only show significant repos.",
        )
        return parser

    def execute(self) -> List[Dict[str, Any]]:
        """Convenience method which executes this CompareCommand using a number of workers."""
        clients = find_repositories(self.paths, nested=self.nested)
        jobs = generate_jobs(clients, self)
        return execute_jobs(
            jobs,
            show_progress=self.progress,
            number_of_workers=self.workers,
            debug_jobs=self.debug,
        )


class Legend:
    """Namespace containing symbols used in the CompareTable."""

    # fmt: off
    SUBMODULE        = "c"
    MISSING          = "M"
    SUPER_PROJECT    = "s"
    REPO_NOT_TRACKED = "U"
    BEHIND           = "<"
    AHEAD            = ">"
    DIVERGED         = "<>"
    UNSTAGED         = "*"
    STAGED           = "+"
    UNTRACKED        = "%"
    STASHES          = "$"
    # fmt: on

    @classmethod
    def get_symbol_description(cls, symbol: str) -> str:
        """Returns a text description of the specified symbol"""
        # fmt: off
        return {
            cls.SUBMODULE        : f"{symbol} submodule",
            cls.MISSING          : f"{symbol} missing",
            cls.SUPER_PROJECT    : f"{symbol} super project",
            cls.REPO_NOT_TRACKED : f"{symbol} repo not tracked",
            cls.BEHIND           : f"{symbol} behind",
            cls.AHEAD            : f"{symbol} ahead",
            cls.DIVERGED         : f"{symbol} diverged",
            cls.UNSTAGED         : f"{symbol} unstaged",
            cls.STAGED           : f"{symbol} staged",
            cls.UNTRACKED        : f"{symbol} untracked",
            cls.STASHES          : f"{symbol} stashes",
        }[symbol]
        # fmt: on

    class Flags(IntFlag):
        """Flags used to indicate which items in the legend should be displayed."""

        MISSING_REPO = auto()
        VCS_TRACKING_STATUS = auto()
        REPO_STATUS = auto()

    @classmethod
    def get_string(cls, flags: Flags, max_width: int, manifest_file: str) -> str:
        """Returns the legend as a string based on flags."""
        legend = "\n"
        if flags & cls.Flags.VCS_TRACKING_STATUS:
            vcs_tracking_symbols = [
                cls.BEHIND,
                cls.AHEAD,
                cls.DIVERGED,
                cls.UNSTAGED,
                cls.STAGED,
                cls.UNTRACKED,
                cls.STASHES,
            ]
            legend += cls._legend_from_symbols(vcs_tracking_symbols, max_width) + "\n"
        if flags & cls.Flags.REPO_STATUS:
            repo_status_symbols = [
                cls.SUBMODULE,
                cls.MISSING,
                cls.SUPER_PROJECT,
                cls.REPO_NOT_TRACKED,
            ]
            legend += cls._legend_from_symbols(repo_status_symbols, max_width) + "\n"
        if flags & cls.Flags.MISSING_REPO:
            legend += "\n" + cls._missing_repos_tip_str(manifest_file) + "\n"
        # Remove the extra newline if no legend is needed.
        return legend if legend != "\n" else ""

    @classmethod
    def _legend_from_symbols(cls, symbols: List[str], max_width: int) -> str:
        separator = 5 * " "
        legend = separator.join(map(cls.get_symbol_description, symbols))
        if len(legend) < max_width:
            margin_length = int((max_width - len(legend)) / 2)
            legend = (" " * margin_length) + legend
        return ansi("brightblackf") + legend + ansi("reset")

    @staticmethod
    def _missing_repos_tip_str(manifest_file: str) -> str:
        tip = [
            "Tip: it looks like you have missing repositories. ",
            "To initialize them execute the following commands:\n",
            f"\tvcs import src < {manifest_file}",
        ]
        tip_str = "".join(tip)
        return ansi("brightmagentaf") + tip_str + ansi("reset")


class RepoStatus(IntEnum):
    """Enum indicating the status of the repository."""

    NOMINAL = 0
    UNTRACKED = 1


class VcsTrackingStatus(IntEnum):
    """Enum indicating the tracking status of the repository."""

    EQUAL = 0
    LOCAL = 1
    LOCAL_BEHIND_REMOTE = 2
    LOCAL_AHEAD_OF_REMOTE = 3
    DIVERGED = 4
    ERROR = 5


class ICompareTableEntry(abc.ABC):
    """Interface used to implement an entry (row) in the CompareTable."""

    HEADERS = ["S", "Repository", "Branch", "Trk", "Flags", "Tag", "Hash"]

    def get_color_row(self, is_odd_row: bool, num_cols: int) -> List[str]:
        """Returns a formatted and colored row representing this entry."""
        # The order of these entries should match the order of HEADERS.
        row = [
            self.get_color_repo_status(),
            self.get_color_repository(),
            self.get_color_branch(),
            self.get_color_track(),
            self.get_color_vcs_tracking_flags(),
            self.get_color_tag(),
            self.get_color_hash(),
        ]
        row = self._wrap_row_with_background_color(row, is_odd_row)
        # max_dispaly_cols is used to hide columns if the terminal is not wide enough to display
        # the full table.
        return row[:num_cols]

    @staticmethod
    def _wrap_row_with_background_color(row: List[str], is_odd_row: bool):
        reset = ansi("reset")
        background = ansi("grey4b") if is_odd_row else reset
        return [background + item + reset + background for item in row]

    @abc.abstractmethod
    def is_significant(self) -> bool:
        return NotImplemented

    @abc.abstractmethod
    def legend_flags(self) -> Legend.Flags:
        return NotImplemented

    @abc.abstractmethod
    def get_color_repo_status(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def get_color_repository(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def get_color_branch(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def get_color_track(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def get_color_vcs_tracking_flags(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def get_color_tag(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def get_color_hash(self) -> str:
        return NotImplemented


class CompareTable(pt.PrettyTable):
    """PrettyTable extension which displays a table describing the state of the workspace."""

    def __init__(
        self,
        entries: Dict[str, ICompareTableEntry],
        manifest_file: str,
        max_width: int = os.get_terminal_size().columns,
    ) -> None:
        super().__init__()

        self._entries = entries
        self._manifest_file = manifest_file
        self._legend_flags = Legend.Flags(0)

        # Initial generation.
        num_cols = len(ICompareTableEntry.HEADERS)
        self._reset_and_add_entries(num_cols)

        # If the table width is too wide, continually remove columns from the right.
        # TODO(amusco): More efficient way to do this?
        DISPLAY_WIDTH_MARGIN = 10
        max_width -= DISPLAY_WIDTH_MARGIN
        while self._table_width() >= max_width:
            num_cols -= 1
            self._reset_and_add_entries(num_cols)

    def _reset_and_add_entries(self, num_cols: int) -> None:
        self.clear()
        self._format_table(num_cols)
        self._legend_flags = Legend.Flags(0)
        for path in sorted(self._entries.keys()):
            self._add_entry(self._entries[path], num_cols)

    def _format_table(self, num_cols: int) -> None:
        """Adds the target column names and formatting to the table."""
        # Dummy column for display formatting purposes.
        DUMMY_END_HEADER = [""]
        self.field_names = ICompareTableEntry.HEADERS[:num_cols] + DUMMY_END_HEADER
        # Default left alignment for all headers
        for header in self.field_names:
            self.align[header] = "l"
        self.border = True
        self.hrules = pt.HEADER
        self.vrules = pt.NONE

    def _add_entry(self, entry: ICompareTableEntry, num_cols: int) -> None:
        # Dummy column for display formatting purposes.
        DUMMY_END_ROW = [ansi("reset")]
        self._legend_flags |= entry.legend_flags()
        is_odd_row = (self.rowcount % 2) == 1
        self.add_row(entry.get_color_row(is_odd_row, num_cols) + DUMMY_END_ROW)

    def _table_width(self) -> int:
        # Need to call get_string() so that _compute_table_width() works properly.
        self.get_string()
        return self._compute_table_width(self._get_options({}))

    def __str__(self) -> str:
        return self.get_string() + Legend.get_string(
            flags=self._legend_flags,
            max_width=self._table_width(),
            manifest_file=self._manifest_file,
        )


class MissingManifestEntry(ICompareTableEntry):
    """Entry for a repo which is specified in the manifest but not included in the CompareOutput."""

    def __init__(self, path: str) -> None:
        self._path = path

    def is_significant(self) -> bool:
        return True

    def legend_flags(self) -> Legend.Flags:
        return Legend.Flags.REPO_STATUS | Legend.Flags.MISSING_REPO

    def get_color_repo_status(self) -> str:
        return ansi("redf") + Legend.MISSING

    def get_color_repository(self) -> str:
        return ansi("redf") + self._path

    def get_color_branch(self) -> str:
        return ansi("redf") + "ABSENT FROM FILESYSTEM"

    def get_color_track(self) -> str:
        return ""

    def get_color_vcs_tracking_flags(self) -> str:
        return ""

    def get_color_tag(self) -> str:
        return ""

    def get_color_hash(self) -> str:
        return ""


class CompareOutputEntry(ICompareTableEntry):
    """Entry for a repo which is discovered by the CompareCommand, but may be missing from the
    manifest (i.e. `manifest_version is None`)"""

    def __init__(
        self,
        path: str,
        compare_output: CompareOutput,
        manifest_version: Optional[str] = None,
    ) -> None:
        self._path = path
        self._compare_output = compare_output
        self._manifest_version = manifest_version
        self._is_current_with_manifest = self._manifest_version in [
            self._compare_output.local_branch,
            self._compare_output.hash,
        ]

        self._vcs_tracking_flags = self._get_vcs_tracking_flags()
        self._repo_status = self._get_repo_status()
        self._vcs_tracking_status = self._get_tracking_status()

    def is_significant(self) -> bool:
        return any(
            [
                self._is_dirty(),
                self._vcs_tracking_status != VcsTrackingStatus.EQUAL,
                # TODO(amusco): is empty git repo?
            ]
        )

    def legend_flags(self) -> Legend.Flags:
        flags = Legend.Flags(0)
        if (
            self._vcs_tracking_status != VcsTrackingStatus.EQUAL
            or self._vcs_tracking_flags.strip() != ""
        ):
            flags |= Legend.Flags.VCS_TRACKING_STATUS
        if self._repo_status != RepoStatus.NOMINAL:
            flags |= Legend.Flags.REPO_STATUS
        return flags

    def get_color_repo_status(self) -> str:
        return {
            RepoStatus.NOMINAL: ansi("brightblackf") + Legend.SUBMODULE,
            RepoStatus.UNTRACKED: ansi("brightyellowf") + Legend.REPO_NOT_TRACKED,
        }[self._repo_status]

    def get_color_repository(self) -> str:
        foreground = ansi("brightcyanf") if self.is_significant() else ansi("whitef")
        return foreground + self._path

    def get_color_branch(self) -> str:
        foreground = ansi("white")
        if not self._is_valid_branch_name(self._compare_output.local_branch):
            foreground = ansi("redf")
        elif self._vcs_tracking_status not in [
            VcsTrackingStatus.EQUAL,
            VcsTrackingStatus.LOCAL,
        ]:
            foreground = ansi("redf")
        elif self._is_current_with_manifest:
            foreground = ansi("brightblackf")
        return foreground + self._compare_output.local_branch

    def get_color_track(self) -> str:
        yellow = ansi("brightyellowf")
        return {
            VcsTrackingStatus.EQUAL: ansi("brightblackf") + "eq",
            VcsTrackingStatus.LOCAL: ansi("whitef") + "local",
            VcsTrackingStatus.LOCAL_BEHIND_REMOTE: (
                yellow + f"{Legend.BEHIND}({self._compare_output.behind})"
            ),
            VcsTrackingStatus.LOCAL_AHEAD_OF_REMOTE: (
                yellow + f"{Legend.AHEAD}({self._compare_output.ahead})"
            ),
            VcsTrackingStatus.DIVERGED: (
                yellow
                + f"{Legend.DIVERGED}({self._compare_output.ahead}, {self._compare_output.behind})"
            ),
            VcsTrackingStatus.ERROR: ansi("redf") + "ERR",
        }[self._vcs_tracking_status]

    def get_color_vcs_tracking_flags(self) -> str:
        foreground = ansi("redf") if self._is_dirty() else ansi("whitef")
        return foreground + self._vcs_tracking_flags

    def get_color_tag(self) -> str:
        foreground = ansi("brightmagentaf")
        return foreground + self._compare_output.tag

    def get_color_hash(self) -> str:
        foreground = ansi("brightblackf")
        return foreground + self._compare_output.hash

    def _get_repo_status(self) -> None:
        if self._manifest_version is None:
            return RepoStatus.UNTRACKED
        return RepoStatus.NOMINAL

    def _get_vcs_tracking_flags(self) -> str:
        flags = ""
        flags += f"{Legend.UNSTAGED}" if self._compare_output.unstaged_changes else " "
        flags += f"{Legend.STAGED}" if self._compare_output.staged_changes else " "
        flags += f"{Legend.UNTRACKED}" if self._compare_output.untracked_files else " "
        flags += f"{Legend.STASHES}" if self._compare_output.stashes else " "
        return flags

    def _is_dirty(self) -> bool:
        # Note: stashes do not count towards 'dirtiness'.
        return any(
            [
                self._compare_output.unstaged_changes,
                self._compare_output.staged_changes,
                self._compare_output.untracked_files,
            ]
        )

    @staticmethod
    def _is_valid_branch_name(branch_name) -> bool:
        acceptable_full = ("master", "develop", "azevtec", "outrider")
        if branch_name in acceptable_full:
            return True
        prefixes = (
            "bugfix/",
            "demo/",
            "feature/",
            "hotfix/",
            "int/",
            "pilot/",
            "release/",
        )
        for prefix in prefixes:
            if branch_name.startswith(prefix):
                return True
        return False

    def _get_tracking_status(self) -> VcsTrackingStatus:
        if not self._compare_output.remote_branch:
            return VcsTrackingStatus.LOCAL
        ahead, behind = self._compare_output.ahead, self._compare_output.behind
        if ahead == 0 and behind == 0:
            return VcsTrackingStatus.EQUAL
        if ahead == 0 and behind > 0:
            return VcsTrackingStatus.LOCAL_BEHIND_REMOTE
        if ahead > 0 and behind == 0:
            return VcsTrackingStatus.LOCAL_AHEAD_OF_REMOTE
        if ahead > 0 and behind > 0:
            return VcsTrackingStatus.DIVERGED
        return VcsTrackingStatus.ERROR


def get_manifest_version(
    manifest_per_path: Dict[str, Dict[str, str]], path: str
) -> str:
    if path not in manifest_per_path:
        return None
    return manifest_per_path[path].get("version", None)


def generate_table_entries(
    compare_output_per_path: Dict[str, CompareOutput],
    manifest_per_path: Dict[str, Dict[str, str]],
    significant_only: bool,
) -> Dict[str, ICompareTableEntry]:
    """Generates a dict of table entries using the output of the CompareCommand and the parsed
    manifest file."""

    entries: Dict[str, ICompareTableEntry] = {}
    # Add entries found in the CompareCommand (but may be mssing from the manifest).
    for path, compare_output in compare_output_per_path.items():
        manifest_version = get_manifest_version(manifest_per_path, path)
        entry = CompareOutputEntry(path, compare_output, manifest_version)
        if significant_only and not entry.is_significant():
            continue
        entries[path] = entry
    # Add entries which exist in the manifest but are missing from the filesystem.
    compare_output_paths = set(compare_output_per_path.keys())
    for path in manifest_per_path:
        if path not in compare_output_paths:
            entries[path] = MissingManifestEntry(path)
    return entries


def print_err(msg: str):
    print(ansi("redf") + msg + ansi("reset"), file=sys.stderr)


def read_repos_from_manifest_file(manifest_file: str) -> Dict[str, Dict[str, Any]]:
    try:
        with open(manifest_file, "r", encoding="utf-8") as infile:
            return get_repositories(infile)
    except RuntimeError as ex:
        print_err(str(ex))
    return None


def filter_compare_output_per_path(
    results: List[Dict[str, Any]], root_path: str
) -> Dict[str, CompareOutput]:
    """Removes entries in results which failed and extracts the CompareOutput from the 'output'
    attribute, returning a dict of output_per_path."""
    output_per_path: Dict[str, CompareOutput] = {}
    for result in results:
        path = result["cwd"].replace(f"{root_path}/", "")
        output = result["output"]
        if result["returncode"] != 0:
            print_err(f"Compare command failed for repo {path}: {output}")
            continue
        assert isinstance(output, CompareOutput)
        output_per_path[path] = output
    return output_per_path


def main(args=None, stdout=None, stderr=None) -> None:
    set_streams(stdout=stdout, stderr=stderr)

    parser = CompareCommand.get_parser()
    add_common_arguments(
        parser, skip_hide_empty=True, skip_nested=True, skip_repos=True, path_nargs="?"
    )
    args = parser.parse_args(args)

    manifest_per_path = read_repos_from_manifest_file(manifest_file=args.input)
    if manifest_per_path is None:
        return 1

    result = CompareCommand(args).execute()
    compare_output_per_path = filter_compare_output_per_path(
        result, root_path=args.path
    )
    if len(compare_output_per_path) == 0:
        return 1

    entries = generate_table_entries(
        compare_output_per_path,
        manifest_per_path,
        significant_only=args.significant,
    )
    print(CompareTable(entries, manifest_file=args.input))
    return 0


if __name__ == "__main__":
    sys.exit(main())
