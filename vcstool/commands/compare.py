import argparse
import sys
import glob
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

# Configuration constants
HASH_MAX_LENGTH = 7
VERSION_MAX_LENGTH = 35
DISPLAY_WIDTH_MARGIN = 8


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
            default="",
            help="Where to read YAML file from.",
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
        group.add_argument(
            "--skip-hide-cols",
            action="store_true",
            default=False,
            help="Don't hide columns in the table to fit the display.",
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


class Colors:
    """Namespace containing colors used throughout the CompareTable"""

    # fmt: off
    RESET                    = ansi("reset")
    ROW_BACKGROUND_ODD       = ansi("reset")
    ROW_BACKGROUND_EVEN      = ansi("grey4b")
    LEGEND                   = ansi("brightblackf")
    TAG                      = ansi("brightmagentaf")
    TIP                      = ansi("brightmagentaf")
    MISSING_REPO             = ansi("redf")
    REPO_STATUS_NOMINAL      = ansi("brightblackf")
    REPO_STATUS_UNTRACKED    = ansi("brightyellowf")
    REPO_STATUS_DIRTY        = ansi("brightyellowf")
    REPO_STATUS_CLEAN        = ""
    SIGNIFICANT              = ansi("brightcyanf")
    INVALID_BRANCH_NAME      = ansi("redf")
    VCS_TRACKING_DIFFERENT   = ansi("brightyellowf")
    VCS_TRACKING_CURRENT     = ansi("brightblackf")
    VCS_TRACKING_BEHIND      = ansi("brightyellowf")
    VCS_TRACKING_AHEAD       = ansi("brightyellowf")
    VCS_TRACKING_DIVERGED    = ansi("brightyellowf")
    VCS_TRACKING_EQUAL       = ansi("brightblackf")
    VCS_TRACKING_LOCAL       = ansi("brightblackf")
    MANIFEST_VERSION_NOMINAL = ansi("brightblackf")
    ERROR                    = ansi("redf")
    # fmt: on


class Legend:
    """Namespace containing symbols used in the CompareTable."""

    # fmt: off
    REPO             = "r"
    MISSING          = "M"
    SUPER_PROJECT    = "s"
    REPO_NOT_TRACKED = "U"
    BEHIND           = ">"
    AHEAD            = "<"
    DIVERGED         = "<>"
    EQUAL            = "=="
    LOCAL            = ""
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
            cls.REPO             : f"{symbol} repository",
            cls.MISSING          : f"{symbol} missing",
            cls.SUPER_PROJECT    : f"{symbol} super project",
            cls.REPO_NOT_TRACKED : f"{symbol} repo not tracked",
            cls.BEHIND           : f"{symbol} behind",
            cls.AHEAD            : f"{symbol} ahead",
            cls.DIVERGED         : f"{symbol} diverged",
            cls.EQUAL            : f"{symbol} equal",
            cls.LOCAL            : f"{symbol} local",
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
        IS_NARROWED = auto()

    @classmethod
    def get_string(cls, flags: Flags, display_width: int, manifest_file: str) -> str:
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
            legend += (
                cls._legend_from_symbols(vcs_tracking_symbols, display_width) + "\n"
            )
        if flags & cls.Flags.REPO_STATUS:
            repo_status_symbols = [
                cls.REPO,
                cls.MISSING,
                cls.SUPER_PROJECT,
                cls.REPO_NOT_TRACKED,
            ]
            legend += (
                cls._legend_from_symbols(repo_status_symbols, display_width) + "\n"
            )
        if flags & cls.Flags.MISSING_REPO:
            legend += "\n" + cls._missing_repos_tip_str(manifest_file) + "\n"
        if flags & cls.Flags.IS_NARROWED:
            legend += "\n" + cls._hidden_cols_tip_str() + "\n"
        # Remove the extra newline if no legend is needed.
        return legend if legend != "\n" else ""

    @classmethod
    def _legend_from_symbols(cls, symbols: List[str], display_width: int) -> str:
        separator = 5 * " "
        legend = separator.join(map(cls.get_symbol_description, symbols))
        if len(legend) < display_width:
            margin_length = int((display_width - len(legend)) / 2)
            legend = (" " * margin_length) + legend
        return Colors.LEGEND + legend + Colors.RESET

    @classmethod
    def _missing_repos_tip_str(cls, manifest_file: str) -> str:
        return cls._format_tip_str(
            "Tip: it looks like you have missing repositories. "
            + "To initialize them execute the following commands:\n"
            + f"\tvcs import src < {manifest_file}",
        )

    @classmethod
    def _hidden_cols_tip_str(cls) -> str:
        return cls._format_tip_str(
            "Tip: some columns have been hidden to fit this display. To view the full table, try "
            + "increasing the character width of this display, or run with the --skip-hide-cols flag."
        )

    @staticmethod
    def _format_tip_str(tip_str: str) -> str:
        return Colors.TIP + tip_str + Colors.RESET


class RepoStatus(IntEnum):
    """Enum indicating the status of the repository."""

    NOMINAL = 0
    UNTRACKED = 1


class VcsTrackingStatus(IntEnum):
    """Enum indicating the tracking status of the repository."""

    EQUAL = 0
    LOCAL = 1
    BEHIND = 2
    AHEAD = 3
    DIVERGED = 4
    ERROR = 5


class ICompareTableEntry(abc.ABC):
    """Interface used to implement an entry (row) in the CompareTable."""

    STATUS_HEADER = "S"
    PATH_HEADER = "Path"
    FLAGS_HEADER = "Flags"
    MANIFEST_VERSION_HEADER = "Manifest"
    LOCAL_VERSION_HEADER = "Local Version"
    TRACKING_STATUS_HEADER = "Ah/Bh"
    REMOTE_VERSION_HEADER = "Remote Version"

    HEADERS = [
        STATUS_HEADER,
        PATH_HEADER,
        FLAGS_HEADER,
        MANIFEST_VERSION_HEADER,
        LOCAL_VERSION_HEADER,
        TRACKING_STATUS_HEADER,
        REMOTE_VERSION_HEADER,
    ]

    # Order to hide rows in the table when the terminal is too small to display all columns.
    HEADER_HIDE_ORDER = [
        REMOTE_VERSION_HEADER,
        MANIFEST_VERSION_HEADER,
        TRACKING_STATUS_HEADER,
        LOCAL_VERSION_HEADER,
        FLAGS_HEADER,
        STATUS_HEADER,
    ]

    HEADER_ALIGNMENT = {
        STATUS_HEADER: "l",
        PATH_HEADER: "l",
        FLAGS_HEADER: "l",
        MANIFEST_VERSION_HEADER: "l",
        LOCAL_VERSION_HEADER: "l",
        TRACKING_STATUS_HEADER: "c",
        REMOTE_VERSION_HEADER: "l",
    }

    MAX_HIDE_COLUMNS = len(HEADER_HIDE_ORDER)

    def get_color_row(self, is_odd_row: bool, cols_to_hide: int) -> List[str]:
        """Returns a formatted and colored row representing this entry."""
        # The order of these entries should match the order of HEADERS.
        row = [
            self.get_color_repo_status(),
            self.get_color_path(),
            self.get_color_vcs_tracking_flags(),
            self.get_color_manifest_version(),
            self.get_color_local_version(),
            self.get_color_track(),
            self.get_color_remote_version(),
        ]
        row = self._wrap_row_with_background_color(row, is_odd_row)
        return self._hide_n_columns(row, cols_to_hide)

    @classmethod
    def get_headers(cls, cols_to_hide: int) -> List[str]:
        """Returns the headers for the table, displaying only `cols_to_hide`. The order the
        columns are hidden are governed by HEADER_HIDE_ORDER.
        """
        headers = list(cls.HEADERS)
        # Make sure we don't attempt to hide too many.
        assert cols_to_hide <= cls.MAX_HIDE_COLUMNS
        for i in range(cols_to_hide):
            header_to_hide = cls.HEADER_HIDE_ORDER[i]
            idx_of_col_to_hide = headers.index(header_to_hide)
            del headers[idx_of_col_to_hide]
        # Dummy column for display formatting purposes.
        DUMMY_END_HEADER = [""]
        return headers + DUMMY_END_HEADER

    @classmethod
    def _hide_n_columns(cls, row: List[str], cols_to_hide: int) -> List[str]:
        # Remove headers from a local copy so that the indices match.
        headers = list(cls.HEADERS)
        # Make sure we don't attempt to hide too many.
        cols_to_hide = min(len(cls.HEADER_HIDE_ORDER), cols_to_hide)
        for i in range(cols_to_hide):
            header_to_hide = cls.HEADER_HIDE_ORDER[i]
            idx_of_col_to_hide = headers.index(header_to_hide)
            del headers[idx_of_col_to_hide]
            del row[idx_of_col_to_hide]
        # Dummy column for display formatting purposes.
        DUMMY_END_ROW = [Colors.RESET]
        return row + DUMMY_END_ROW

    @staticmethod
    def _wrap_row_with_background_color(row: List[str], is_odd_row: bool):
        background = (
            Colors.ROW_BACKGROUND_ODD if is_odd_row else Colors.ROW_BACKGROUND_EVEN
        )
        return [background + item + Colors.RESET + background for item in row]

    @abc.abstractmethod
    def is_significant(self) -> bool:
        return NotImplemented

    @abc.abstractmethod
    def get_legend_flags(self) -> Legend.Flags:
        return NotImplemented

    @abc.abstractmethod
    def get_color_repo_status(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def get_color_path(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def get_color_local_version(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def get_color_remote_version(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def get_color_remote_hash(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def get_color_manifest_version(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def get_color_track(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def get_color_vcs_tracking_flags(self) -> str:
        return NotImplemented


class CompareTable(pt.PrettyTable):
    """PrettyTable extension which displays a table describing the state of the workspace."""

    def __init__(
        self,
        entries: Dict[str, ICompareTableEntry],
        manifest_file: str,
        should_hide_cols: bool = False,
        display_width: int = os.get_terminal_size().columns - DISPLAY_WIDTH_MARGIN,
    ) -> None:
        super().__init__()
        self._entries = entries
        self._manifest_file = manifest_file
        self._legend_flags = Legend.Flags(0)
        self._display_width = display_width
        self._should_hide_cols = should_hide_cols
        self._cols_to_hide = 0

        self._reset_and_add_entries()
        if self._is_narrow_enough():
            return

        # If table is too wide, try abbreviating the version names.
        for _, entry in self._entries.items():
            entry.should_abbreviate_version = True
        self._reset_and_add_entries()
        if self._is_narrow_enough():
            return

        # If that didn't work, hide the columns until we're narrow enough or can't hide any more
        # columns.
        while not self._is_narrow_enough() and self._can_hide_more_cols():
            self._cols_to_hide += 1
            self._reset_and_add_entries()

    def _is_narrow_enough(self) -> bool:
        return not self._should_hide_cols or (
            self._table_width() <= self._display_width
        )

    def _can_hide_more_cols(self) -> bool:
        return self._cols_to_hide < ICompareTableEntry.MAX_HIDE_COLUMNS

    def _reset_and_add_entries(self) -> None:
        self.clear()
        self._format_table()
        self._legend_flags = (
            Legend.Flags.IS_NARROWED if self._cols_to_hide > 0 else Legend.Flags(0)
        )
        for path in sorted(self._entries.keys()):
            self._add_entry(self._entries[path])

    def _format_table(self) -> None:
        """Adds the target column names and formatting to the table."""
        self.field_names = ICompareTableEntry.get_headers(self._cols_to_hide)
        for header, align in ICompareTableEntry.HEADER_ALIGNMENT.items():
            self.align[header] = align
        self.border = True
        self.hrules = pt.HEADER
        self.vrules = pt.NONE

    def _add_entry(self, entry: ICompareTableEntry) -> None:
        is_odd_row = (self.rowcount % 2) == 1
        self._legend_flags |= entry.get_legend_flags()
        self.add_row(entry.get_color_row(is_odd_row, self._cols_to_hide))

    def _table_width(self) -> int:
        # Need to call get_string() so that _compute_table_width() works properly.
        self.get_string()
        return self._compute_table_width(self._get_options({}))

    def __str__(self) -> str:
        # If we have no rows, don't display anything
        if self.rowcount == 0:
            return ""
        return self.get_string() + Legend.get_string(
            flags=self._legend_flags,
            display_width=self._table_width(),
            manifest_file=self._manifest_file,
        )


class MissingManifestEntry(ICompareTableEntry):
    """Entry for a repo which is specified in the manifest but not included in the CompareOutput."""

    def __init__(self, path: str, manifest_version: str) -> None:
        self._path = path
        self._manifest_version = manifest_version

    def is_significant(self) -> bool:
        return True

    def get_legend_flags(self) -> Legend.Flags:
        return Legend.Flags.REPO_STATUS | Legend.Flags.MISSING_REPO

    def get_color_repo_status(self) -> str:
        return Colors.MISSING_REPO + Legend.MISSING

    def get_color_path(self) -> str:
        return Colors.MISSING_REPO + self._path

    def get_color_local_version(self) -> str:
        return ""

    def get_color_remote_version(self) -> str:
        return ""

    def get_color_remote_hash(self) -> str:
        return ""

    def get_color_manifest_version(self) -> str:
        return self._manifest_version

    def get_color_track(self) -> str:
        return ""

    def get_color_vcs_tracking_flags(self) -> str:
        return ""


def is_probably_a_hash(the_hash: str) -> bool:
    """Returns true if the string is a hash... probably."""
    HASH_NUM_CHARACTERS = 40
    return len(the_hash) == HASH_NUM_CHARACTERS


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
        self._compare_output.fix_detached_head()
        self._manifest_version = manifest_version
        self._is_local_current_with_manifest = self._manifest_version in [
            self._compare_output.local_version,
            self._compare_output.local_hash,
        ]
        self._is_remote_current_with_manifest = self._manifest_version in [
            self._compare_output.remote_version,
            self._compare_output.remote_hash,
        ]

        self._vcs_tracking_flags = self._get_vcs_tracking_flags()
        self._repo_status = self._get_repo_status()
        self._vcs_tracking_status = self._get_tracking_status()

        self.should_abbreviate_version = False

    def is_significant(self) -> bool:
        return any(
            [
                self._is_dirty(),
                self._vcs_tracking_status != VcsTrackingStatus.EQUAL,
                self._repo_status != RepoStatus.NOMINAL,
                self._manifest_version != self._compare_output.local_version,
            ]
        )

    def get_legend_flags(self) -> Legend.Flags:
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
            RepoStatus.NOMINAL: Colors.REPO_STATUS_NOMINAL + Legend.REPO,
            RepoStatus.UNTRACKED: (
                Colors.REPO_STATUS_UNTRACKED + Legend.REPO_NOT_TRACKED
            ),
        }[self._repo_status]

    def get_color_path(self) -> str:
        foreground = Colors.SIGNIFICANT if self.is_significant() else ""
        return f"{foreground}{self._path}"

    def _get_abbreviated_version(self, version: str, max_len: int = VERSION_MAX_LENGTH):
        if self.should_abbreviate_version is False:
            return version
        is_short_enough = len(version) <= max_len
        ELLIPSIS = "..."
        truncate_idx = max_len - len(ELLIPSIS)
        return version if is_short_enough else version[:truncate_idx] + ELLIPSIS

    @classmethod
    def _get_abbreviated_hash(cls, the_hash: str, max_len: int = HASH_MAX_LENGTH):
        if is_probably_a_hash(the_hash) == False:
            return the_hash
        is_short_enough = len(the_hash) <= max_len
        return the_hash if is_short_enough else the_hash[:max_len]

    def _get_local_foreground_color(self) -> str:
        if not self._is_valid_branch_name(self._compare_output.local_version):
            return Colors.INVALID_BRANCH_NAME
        elif self._vcs_tracking_status not in [
            VcsTrackingStatus.EQUAL,
            VcsTrackingStatus.LOCAL,
        ]:
            return Colors.VCS_TRACKING_DIFFERENT
        elif self._is_local_current_with_manifest:
            return Colors.VCS_TRACKING_CURRENT
        return ""

    def get_color_local_version(self) -> str:
        foreground = self._get_local_foreground_color()
        local_version = self._get_abbreviated_version(
            self._compare_output.local_version
        )
        local_hash = self._get_abbreviated_hash(self._compare_output.local_hash)
        local_hash = f"{local_hash}" if local_hash != "" else ""
        return f"{foreground}{local_hash} ({local_version})"

    def _get_remote_foreground_color(self) -> str:
        if self._vcs_tracking_status not in [
            VcsTrackingStatus.EQUAL,
            VcsTrackingStatus.LOCAL,
        ]:
            return Colors.VCS_TRACKING_DIFFERENT
        elif self._is_remote_current_with_manifest:
            return Colors.VCS_TRACKING_CURRENT
        return ""

    def get_color_remote_version(self) -> str:
        remote = self._compare_output.remote
        remote_version = self._get_abbreviated_version(
            self._compare_output.remote_version
        )
        if remote == "" or remote_version == "":
            return ""
        foreground = self._get_remote_foreground_color()
        remote_hash = self._get_abbreviated_hash(self._compare_output.remote_hash)
        remote_hash = f"{remote_hash}" if remote_hash != "" else ""
        return f"{foreground}{remote_hash} ({remote}/{remote_version})"

    def get_color_remote_hash(self) -> str:
        remote_hash = self._get_abbreviated_hash(self._compare_output.remote_hash)
        if remote_hash == "":
            return ""
        foreground = self._get_remote_foreground_color()
        return f"{foreground}{remote_hash}"

    def get_color_manifest_version(self) -> str:
        if self._manifest_version is None:
            return ""
        manifest_version = self._manifest_version
        if is_probably_a_hash(manifest_version):
            manifest_version = f"{self._get_abbreviated_hash(manifest_version)}"
        foreground = ""
        if self._is_local_current_with_manifest:
            foreground = Colors.MANIFEST_VERSION_NOMINAL
        return f"{foreground}{manifest_version}"

    def get_color_track(self) -> str:
        ahead, behind = self._compare_output.ahead, self._compare_output.behind
        return {
            VcsTrackingStatus.EQUAL: f"{Colors.VCS_TRACKING_EQUAL}{Legend.EQUAL}",
            VcsTrackingStatus.LOCAL: f"{Colors.VCS_TRACKING_LOCAL}{Legend.LOCAL}",
            VcsTrackingStatus.AHEAD: (
                f"{Colors.VCS_TRACKING_BEHIND}{Legend.AHEAD}{ahead}"
            ),
            VcsTrackingStatus.BEHIND: (
                f"{Colors.VCS_TRACKING_BEHIND}   {behind}{Legend.BEHIND}"
            ),
            VcsTrackingStatus.DIVERGED: (
                f"{Colors.VCS_TRACKING_DIVERGED}{Legend.AHEAD}{ahead},{behind}{Legend.BEHIND}"
            ),
            VcsTrackingStatus.ERROR: Colors.ERROR + "ERR",
        }[self._vcs_tracking_status]

    def get_color_vcs_tracking_flags(self) -> str:
        foreground = (
            Colors.REPO_STATUS_DIRTY if self._is_dirty() else Colors.REPO_STATUS_CLEAN
        )
        return f"{foreground}{self._vcs_tracking_flags}"

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
        if not self._compare_output.remote_version:
            return VcsTrackingStatus.LOCAL
        ahead, behind = self._compare_output.ahead, self._compare_output.behind
        if ahead == 0 and behind == 0:
            return VcsTrackingStatus.EQUAL
        if ahead == 0 and behind > 0:
            return VcsTrackingStatus.BEHIND
        if ahead > 0 and behind == 0:
            return VcsTrackingStatus.AHEAD
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
    for path, manifest in manifest_per_path.items():
        if path not in compare_output_paths:
            entries[path] = MissingManifestEntry(path, manifest["version"])
    return entries


def print_err(msg: str):
    print(Colors.ERROR + msg + Colors.RESET, file=sys.stderr)


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


def get_manifest_file(manifest_file_in: str):
    if manifest_file_in != "":
        return manifest_file_in
    matches = glob.glob("*.repos")
    if len(matches) == 1:
        return matches[0]
    print_err(
        f"Multiple possible manifest files: {matches}. Please specify one using the --input flag."
    )
    return None


def main(args=None, stdout=None, stderr=None) -> None:
    set_streams(stdout=stdout, stderr=stderr)

    parser = CompareCommand.get_parser()
    add_common_arguments(
        parser, skip_hide_empty=True, skip_nested=True, skip_repos=True, path_nargs="?"
    )
    args = parser.parse_args(args)

    manifest_file = get_manifest_file(args.input)
    if manifest_file is None:
        return 1
    if not os.path.exists(manifest_file):
        print_err(f"Manifest file does not exist: {manifest_file}")

    manifest_per_path = read_repos_from_manifest_file(manifest_file)
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
    print(
        CompareTable(entries, manifest_file, should_hide_cols=not args.skip_hide_cols)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
