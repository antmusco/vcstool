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
from vcstool.results import CompareResults

from .command import add_common_arguments
from .command import Command


class CompareCommand(Command):

    command = "compare"
    help = "Compare working copy to the repository list file"

    def __init__(self, args: Dict[str, Any]) -> None:
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
            help="Path to the .repos file for the workspace.",
        )
        group.add_argument(
            "-p", "--progress", action="store_true", default=False, help="Show progress"
        )
        group.add_argument(
            "-n",
            "--nested",
            action="store_true",
            default=False,
            help="Search for nested repositories",
        )
        group.add_argument(
            "-s",
            "--significant",
            action="store_true",
            default=False,
            help="Only show significant repos",
        )
        return parser

    def execute(self) -> Dict[str, Any]:
        """Convenience method which executes this CompareCommand using a number of workers."""
        clients = find_repositories(self.paths, nested=self.nested)
        jobs = generate_jobs(clients, self)
        return execute_jobs(
            jobs,
            show_progress=self.progress,
            number_of_workers=self.workers,
            debug_jobs=self.debug,
        )


class LegendFlags(IntFlag):
    """Flags used to indicate which items in the legend should be displayed."""

    MISSING_REPO = auto()
    TRACKING_AND_FLAGS = auto()
    REPO_STATUS = auto()


class Status(IntEnum):
    """Enum indicating the status of the repository."""

    NOMINAL = 0
    UNTRACKED = 1


class Tracking(IntEnum):
    """Enum indicating the tracking status of the repository."""

    EQUAL = 0
    LOCAL = 1
    LOCAL_BEHIND_REMOTE = 2
    LOCAL_AHEAD_OF_REMOTE = 3
    DIVERGED = 4
    ERR = 5


class IRepoTableEntry(abc.ABC):
    """Interface used to implement an entry (row) in the RepoTable."""

    HEADERS = ["S", "Repository", "Branch", "Trk", "Flags", "Tag", "Hash"]

    def get_color_row(self, is_odd_row: bool, max_display_cols: int) -> List[str]:
        """Returns a formatted and colored row representing this entry."""
        # The order of these entries should match the order of HEADERS.
        cells = [
            self.get_color_status(),
            self.get_color_repository(),
            self.get_color_branch(),
            self.get_color_track(),
            self.get_color_flags(),
            self.get_color_tag(),
            self.get_color_hash(),
        ]
        cells = self._wrap_cells_with_background_color(cells, is_odd_row)
        # max_dispaly_cols is used to hide columns if the terminal is not wide enough to display
        # the full table.
        return cells[:max_display_cols]

    @staticmethod
    def _wrap_cells_with_background_color(cells: List[str], is_odd_row: bool):
        reset = ansi("reset")
        background = ansi("grey4b") if is_odd_row else reset
        return [background + cell + reset + background for cell in cells]

    @abc.abstractmethod
    def is_significant(self) -> bool:
        return NotImplemented

    @abc.abstractmethod
    def legend_flags(self) -> LegendFlags:
        return NotImplemented

    @abc.abstractmethod
    def get_color_status(self) -> str:
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
    def get_color_flags(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def get_color_tag(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def get_color_hash(self) -> str:
        return NotImplemented


class RepoTable(pt.PrettyTable):
    """PrettyTable extension which displays various useful information related to repositories in
    the workspace."""

    # Dummy column for display formatting purposes.
    HEADER_END = [""]
    ROW_END = [ansi("reset")]

    DISPLAY_WIDTH_MARGIN = 10

    def __init__(self, entries: Dict[str, IRepoTableEntry], manifest_file: str) -> None:
        super().__init__()
        self._entries = entries
        self._manifest_file = manifest_file
        self._sorted_paths = sorted(entries.keys())
        self._max_display_cols = len(IRepoTableEntry.HEADERS)
        self._legend_flags: LegendFlags
        # Initial table generation:
        self._reset_and_add_entries()

    def _reset_and_add_entries(self) -> None:
        self.clear()
        self._format_table()
        self._legend_flags = LegendFlags(0)
        for path in self._sorted_paths:
            self._add_entry(self._entries[path])

    def _add_entry(self, entry: IRepoTableEntry) -> None:
        self._legend_flags |= entry.legend_flags()
        is_odd_row = (self.rowcount % 2) == 1
        self.add_row(
            entry.get_color_row(is_odd_row, self._max_display_cols) + self.ROW_END
        )

    def _format_table(self) -> None:
        """Adds the target column names and formatting to the table."""
        self.field_names = (
            IRepoTableEntry.HEADERS[: self._max_display_cols] + self.HEADER_END
        )
        # Default left alignment for all headers
        for header in self.field_names:
            self.align[header] = "l"
        self.border = True
        self.hrules = pt.HEADER
        self.vrules = pt.NONE

    def __str__(self) -> str:
        string = self._table_str()
        if self._legend_flags & LegendFlags.TRACKING_AND_FLAGS:
            string += "\n\n" + self._tracking_and_flags_legend_str()
        if self._legend_flags & LegendFlags.REPO_STATUS:
            string += "\n" + self._repo_status_and_legend_str()
        if self._legend_flags & LegendFlags.MISSING_REPO:
            string += "\n\n" + self._missing_repos_tip_str() + "\n"
        return string

    def _table_str(self) -> None:
        max_width = os.get_terminal_size().columns - self.DISPLAY_WIDTH_MARGIN
        string = self.get_string()
        while self._get_table_width() >= max_width:
            # If the table width is too wide, continually remove columns from the right.
            self._max_display_cols -= 1
            self._reset_and_add_entries()
            string = self.get_string()
        return string

    def _tracking_and_flags_legend_str(self) -> str:
        separator = 5 * " "
        legend = [
            "< behind",
            "> ahead",
            "<> diverged",
            "* unstaged",
            "+ staged",
            "% untracked",
            "$ stashes",
        ]
        legend_str = separator.join(legend)
        table_width = self._get_table_width()
        if table_width > len(legend_str):
            legend_str = " " * int((table_width - len(legend_str)) / 2) + legend_str
        return ansi("brightblackf") + legend_str + ansi("reset")

    def _repo_status_and_legend_str(self) -> str:
        separator = 5 * " "
        legend = [
            "c submodule",
            "M missing",
            "s super project",
            "U not tracked",
        ]
        legend_str = separator.join(legend)
        table_width = self._get_table_width()
        if table_width > len(legend_str):
            legend_str = " " * int((table_width - len(legend_str)) / 2) + legend_str
        return ansi("brightblackf") + legend_str + ansi("reset")

    def _missing_repos_tip_str(self) -> str:
        tip = [
            "Tip: it looks like you have missing repositories. ",
            "To initialize them execute the following commands:\n",
            f"\tvcs import src < {self._manifest_file}",
        ]
        tip_str = "".join(tip)
        return ansi("brightmagentaf") + tip_str + ansi("reset")

    def _get_table_width(self) -> int:
        return self._compute_table_width(self._get_options({}))


class MissingRepoTableEntry(IRepoTableEntry):
    """Entry for a repo which is specified in the manifest but missing from the filesystem."""

    def __init__(self, path: str) -> None:
        self._path = path

    def is_significant(self) -> bool:
        return True

    def legend_flags(self) -> LegendFlags:
        return LegendFlags.REPO_STATUS | LegendFlags.MISSING_REPO

    def get_color_status(self) -> str:
        return ansi("redf") + "D"

    def get_color_repository(self) -> str:
        return ansi("redf") + self._path

    def get_color_branch(self) -> str:
        return ansi("redf") + "ABSENT FROM FILESYSTEM"

    def get_color_track(self) -> str:
        return ""

    def get_color_flags(self) -> str:
        return ""

    def get_color_tag(self) -> str:
        return ""

    def get_color_hash(self) -> str:
        return ""


class ExistingRepoTableEntry(IRepoTableEntry):
    """Entry for a repo which is discovered on within the workspace."""

    def __init__(
        self, path: str, results: CompareResults, manifest_branch: Optional[str]
    ) -> None:
        self._path = path
        self._results = results
        self._manifest_branch = manifest_branch
        self._is_current = self._results.local_branch == self._manifest_branch

        self._flags = self._get_flags()
        self._status = self._get_update_status()
        self._tracking = self._get_tracking_status()

    def is_significant(self) -> bool:
        return any(
            [
                self._is_dirty(),
                # is empty git repo?
                # is in manifest but not present?
                self._tracking != Tracking.EQUAL,
            ]
        )

    def legend_flags(self) -> LegendFlags:
        flags = LegendFlags(0)
        if self._tracking != Tracking.EQUAL:
            flags = flags | LegendFlags.TRACKING_AND_FLAGS
        if self._flags.strip() != "":
            flags = flags | LegendFlags.TRACKING_AND_FLAGS
        if self._status != Status.NOMINAL:
            flags = flags | LegendFlags.REPO_STATUS
        return flags

    def get_color_status(self) -> str:
        return {
            Status.NOMINAL: ansi("brightblackf") + "c",
            Status.UNTRACKED: ansi("brightyellowf") + "U",
        }[self._status]

    def get_color_repository(self) -> str:
        foreground = ansi("brightcyanf") if self.is_significant() else ansi("whitef")
        return foreground + self._path

    def get_color_branch(self) -> str:
        foreground = ansi("white")
        if not self._is_valid_branch_name(self._results.local_branch):
            foreground = ansi("redf")
        elif self._tracking not in [Tracking.EQUAL, Tracking.LOCAL]:
            foreground = ansi("redf")
        elif self._is_current:
            foreground = ansi("brightblackf")
        return foreground + self._results.local_branch

    def get_color_track(self) -> str:
        return {
            Tracking.EQUAL: ansi("brightblackf") + "eq",
            Tracking.LOCAL: ansi("whitef") + "local",
            Tracking.LOCAL_BEHIND_REMOTE: (
                ansi("brightyellowf") + f"<({self._results.behind})"
            ),
            Tracking.LOCAL_AHEAD_OF_REMOTE: (
                ansi("brightyellowf") + f">({self._results.ahead})"
            ),
            Tracking.DIVERGED: (
                ansi("brightyellowf")
                + f"<>({self._results.ahead}, {self._results.behind})"
            ),
            Tracking.ERR: ansi("redf") + "ERR",
        }[self._tracking]

    def get_color_flags(self) -> str:
        foreground = ansi("redf") if self._is_dirty() else ansi("whitef")
        return foreground + self._flags

    def get_color_tag(self) -> str:
        foreground = ansi("brightmagentaf")
        return foreground + self._results.tag

    def get_color_hash(self) -> str:
        foreground = ansi("brightblackf")
        return foreground + self._results.hash

    def _get_update_status(self) -> None:
        if self._manifest_branch is None:
            return Status.UNTRACKED
        return Status.NOMINAL

    def _get_flags(self) -> str:
        flags = ""
        flags += "*" if self._results.unstaged_changes else " "
        flags += "+" if self._results.staged_changes else " "
        flags += "%" if self._results.untracked_files else " "
        flags += "$" if self._results.stashes else " "
        return flags

    def _is_dirty(self) -> bool:
        # Note: stashes do not count towards 'dirtiness'.
        return any(
            [
                self._results.unstaged_changes,
                self._results.staged_changes,
                self._results.untracked_files,
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

    def _get_tracking_status(self) -> Tracking:
        if not self._results.remote_branch:
            return Tracking.LOCAL
        ahead, behind = self._results.ahead, self._results.behind
        if ahead == 0 and behind == 0:
            return Tracking.EQUAL
        if ahead == 0 and behind > 0:
            return Tracking.LOCAL_BEHIND_REMOTE
        if ahead > 0 and behind == 0:
            return Tracking.LOCAL_AHEAD_OF_REMOTE
        if ahead > 0 and behind > 0:
            return Tracking.DIVERGED
        return Tracking.ERR


def get_manifest_branch(manifest_repos: Dict[str, Dict[str, Any]], path: str) -> str:
    if path not in manifest_repos:
        return None
    return manifest_repos[path].get("version", None)


def generate_table_entries(
    compare_results: Dict[str, Any],
    manifest_repos: Dict[str, Dict[str, Any]],
    root_path: str,
    significant_only: bool,
) -> Dict[str, IRepoTableEntry]:
    """Generates a dict of table entries using the output of the CompareCommand and the parsed
    manifest file."""

    entries: Dict[str, IRepoTableEntry] = {}
    existing_paths = set()
    # Add entries found on the filesystem (but may be mssing from the manifest).
    for result in compare_results:
        # Strip the input path from the client path
        path = result["cwd"].replace(f"{root_path}/", "")
        existing_paths.add(path)
        manifest_branch = get_manifest_branch(manifest_repos, path)
        entry = ExistingRepoTableEntry(path, result["output"], manifest_branch)
        if significant_only and not entry.is_significant():
            continue
        entries[path] = entry
    # Add entries which exist in the manifest but are missing from the filesystem.
    for path in manifest_repos:
        if path not in existing_paths:
            entries[path] = MissingRepoTableEntry(path)
    return entries


def read_repos_from_manifest_file(manifest_file_path: str) -> Dict[str, Dict[str, Any]]:
    try:
        with open(manifest_file_path, "r", encoding="utf-8") as manifest_file:
            return get_repositories(manifest_file)
    except RuntimeError as ex:
        print(ansi("redf") + str(ex) + ansi("reset"), file=sys.stderr)
    return None


def main(args=None, stdout=None, stderr=None) -> None:
    set_streams(stdout=stdout, stderr=stderr)

    parser = CompareCommand.get_parser()
    add_common_arguments(
        parser, skip_hide_empty=True, skip_nested=True, skip_repos=True, path_nargs="?"
    )
    args = parser.parse_args(args)

    manifest_file = args.input
    manifest_repos = read_repos_from_manifest_file(manifest_file)
    if manifest_repos is None:
        return 1

    compare_results = CompareCommand(args).execute()
    entries = generate_table_entries(
        compare_results,
        manifest_repos,
        root_path=args.path,
        significant_only=args.significant,
    )
    print(RepoTable(entries, manifest_file))
    return 0


if __name__ == "__main__":
    sys.exit(main())
