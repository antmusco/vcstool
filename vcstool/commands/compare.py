import argparse
import sys
import os
import abc
from typing import (Any, Dict, List, Optional)
from enum import Enum, IntFlag, auto

import prettytable as pt

from vcstool.crawler import find_repositories
from vcstool.executor import ansi, execute_jobs, generate_jobs
from vcstool.commands.import_ import get_repositories
from vcstool.streams import set_streams

from .command import add_common_arguments
from .command import Command


class CompareCommand(Command):

    command = 'compare'
    help = 'Compare working copy to the repository list file'

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)
        self.progress = args.progress
        self.workers = args.workers
        self.debug = args.debug

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=f'{cls.help}', prog='vcs compare')
        group = parser.add_argument_group('"compare" command parameters')
        group.add_argument('-i', '--input', type=str, default='workspace.repos',
                           help='Path to the .repos file for the workspace.')
        group.add_argument('-p', '--progress', action='store_true', default=False,
                           help='Show progress')
        group.add_argument('-n', '--nested', action='store_true', default=False,
                           help='Search for nested repositories')
        group.add_argument('-s', '--significant', action='store_true', default=False,
                           help='Only show significant repos')
        return parser


    def execute(self) -> Dict[str, Any]:
        clients = find_repositories(self.paths, nested=self.nested)
        jobs = generate_jobs(clients, self)
        return execute_jobs(jobs, show_progress=self.progress, number_of_workers=self.workers,
                            debug_jobs=self.debug)


class LegendFlags(IntFlag):
    """Enum for which items in the legend should be printed."""
    MISSING_REPO = auto()
    TRACKING_AND_FLAGS = auto()
    REPO_STATUS = auto()


class Status(Enum):
    """Enum indicating the status of the repository."""
    NOMINAL = 0
    UNTRACKED = 1


class Tracking(Enum):
    """Enum indicating the tracking status of the repository."""
    EQUAL = 0
    LOCAL = 1
    LOCAL_BEHIND_REMOTE = 2
    LOCAL_AHEAD_OF_REMOTE = 3
    DIVERGED = 4
    ERR = 5


class IEntry(abc.ABC):
    """Interface used to implement an entry (row) in the RepoTable."""

    HEADERS = ['S', 'Repository', 'Branch', 'Trk', 'Flags', 'Tag', 'Hash']

    def color_row(self, is_odd: bool, max_display_cols: int) -> List[str]:
        """Returns a formatted and colored row representing this entry."""
        background = ansi('grey4b') if is_odd else ansi('reset')
        reset = ansi('reset')
        # The order of these entries should match the order of HEADERS.
        rows = [
            background + self.color_status() + reset + background,
            background + self.color_path()   + reset + background,
            background + self.color_branch() + reset + background,
            background + self.color_track()  + reset + background,
            background + self.color_flags()  + reset + background,
            background + self.color_tag()    + reset + background,
            background + self.color_hash()    + reset + background,
        ]
        END_COL = [reset]
        # max_dispaly_cols is used to hide columns if the terminal is not wide enough to display
        # the full table.
        return rows[:max_display_cols] + END_COL

    @abc.abstractmethod
    def is_significant(self) -> bool:
        return NotImplemented

    @abc.abstractmethod
    def legend_flags(self) -> LegendFlags:
        return NotImplemented

    @abc.abstractmethod
    def color_status(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def color_path(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def color_branch(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def color_track(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def color_flags(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def color_tag(self) -> str:
        return NotImplemented

    @abc.abstractmethod
    def color_hash(self) -> str:
        return NotImplemented


class RepoTable(pt.PrettyTable):

    def __init__(self, entries: Dict[str, IEntry]) -> None:
        super().__init__()
        self._entries = entries
        self._sorted_paths = sorted(entries.keys())
        self._max_display_cols = len(IEntry.HEADERS)
        self._legend_flags = LegendFlags(0)
        self._generate()

    def print(self, manifest_file: str) -> None:
        self._print_table()
        if self._legend_flags & LegendFlags.TRACKING_AND_FLAGS:
            self._print_tracking_and_flags_legend()
        if self._legend_flags & LegendFlags.REPO_STATUS:
            self._print_repo_status_and_legend()
        if self._legend_flags & LegendFlags.MISSING_REPO:
            self._print_missing_repos(manifest_file=manifest_file)

    def _generate(self) -> None:
        self.clear()
        self._format_table()
        self._legend_flags = LegendFlags(0)
        for path in self._sorted_paths:
            entry = self._entries[path]
            is_odd_row = (self.rowcount % 2) == 1
            self._legend_flags |= entry.legend_flags()
            self.add_row(entry.color_row(is_odd_row, self._max_display_cols))

    def _format_table(self) -> None:
        """Adds the target column names and formatting to the table."""
        DUMMY_END_COL = ['']
        self.field_names = IEntry.HEADERS[:self._max_display_cols] + DUMMY_END_COL
        # Default left alignment for all headers
        for header in self.field_names:
            self.align[header] = "l"
        self.border = True
        self.hrules = pt.HEADER
        self.vrules = pt.NONE

    def _print_table(self) -> None:
        # If the table width is too wide, continually remove columns from the left.
        term_width = os.get_terminal_size().columns
        # Need to call get_string() to ensure get_table_width() works.
        _ = self.get_string()
        MARGIN = 10
        while self._get_table_width() >= term_width - MARGIN:
            self._max_display_cols -= 1
            self._generate()
            _ = self.get_string()
        print(self)

    def _get_table_width(self) -> int:
        return self._compute_table_width(self._get_options({}))

    def _print_tracking_and_flags_legend(self) -> None:
        separator = 5 * ' '
        legend = "< behind" + separator + \
                 "> ahead" + separator + \
                 "<> diverged" + separator + \
                 "* unstaged" + separator + \
                 "+ staged" + separator + \
                 "% untracked" + separator + \
                 "$ stashes"
        table_width = self._get_table_width()
        if table_width > len(legend):
            legend = ' ' * int((table_width - len(legend)) / 2) + legend
        print(ansi('brightblackf') + legend + ansi('reset'))

    def _print_repo_status_and_legend(self) -> None:
        separator = 5 * ' '
        legend = "c submodule" + separator + \
                 "M missing" + separator + \
                 "s super project" + separator + \
                 "U not tracked"
        table_width = self._get_table_width()
        if table_width > len(legend):
            legend = ' ' * int((table_width - len(legend)) / 2) + legend
        print(ansi('brightblackf') + legend + ansi('reset'))

    def _print_missing_repos(self, manifest_file: str) -> None:
        print(ansi('brightmagentaf') +
              "Tip: it looks like you have missing repositories. " +
              "To initialize them execute the following commands:" +
              ansi('reset'))
        print(ansi('brightmagentaf') + f'\tvcs import src < {manifest_file}' + ansi('reset'))


class MissingRepoTableEntry(IEntry):
    """Entry for a repo which is specified in the manifest but missing from the filesystem."""

    def __init__(self, path: str) -> None:
        self._path = path

    def is_significant(self) -> bool:
        return True

    def legend_flags(self) -> LegendFlags:
        return LegendFlags.REPO_STATUS | LegendFlags.MISSING_REPO

    def color_status(self) -> str:
        return ansi('redf') + 'D'

    def color_path(self) -> str:
        return ansi('redf') + self._path

    def color_branch(self) -> str:
        return ansi('redf') + 'ABSENT FROM FILESYSTEM'

    def color_track(self) -> str:
        return ''

    def color_flags(self) -> str:
        return ''

    def color_tag(self) -> str:
        return ''

    def color_hash(self) -> str:
        return ''


class ExistingRepoTableEntry(IEntry):
    """Entry for a repo which is discovered on within the workspace."""

    def __init__(self, path : str, compare : Dict[str, str],
                 manifest_branch : Optional[str]) -> None:
        self._path = path
        self._local_branch = compare['local_branch']
        self._remote_branch = compare['remote_branch']
        self._remote = compare['remote']
        self._ahead = compare['ahead']
        self._behind = compare['behind']
        self._tag = compare['tag']
        self._hash = compare['hash']
        self._are_unstaged_changes = compare['are_unstaged_changes']
        self._are_staged_changes = compare['are_staged_changes']
        self._are_untracked_files = compare['are_untracked_files']
        self._are_stashes = compare['are_stashes']
        self._manifest_branch = manifest_branch
        self._is_current = self._local_branch == self._manifest_branch

        self._flags = self._get_flags()
        self._status = self._get_update_status()
        self._tracking = self._get_tracking_status()

    def is_significant(self) -> bool:
        return any([
            self._is_dirty(),
            # is empty git repo?
            # is in manifest but not present?
            self._tracking != Tracking.EQUAL,
        ])

    def legend_flags(self) -> LegendFlags:
        flags = LegendFlags(0)
        if self._tracking != Tracking.EQUAL:
            flags = flags | LegendFlags.TRACKING_AND_FLAGS
        if self._flags.strip() != '':
            flags = flags | LegendFlags.TRACKING_AND_FLAGS
        if self._status != Status.NOMINAL:
            flags = flags | LegendFlags.REPO_STATUS
        return flags

    def color_status(self) -> str:
        return {
            Status.NOMINAL: ansi('brightblackf') + 'c',
            Status.UNTRACKED: ansi('brightyellowf') + 'U',
        }[self._status]

    def color_path(self) -> str:
        foreground = ansi('brightcyanf') if self.is_significant() else ansi('whitef')
        return foreground + self._path

    def color_branch(self) -> str:
        foreground = ansi('white')
        if not self._is_valid_branch_name(self._local_branch):
            foreground = ansi('redf')
        elif self._tracking not in [Tracking.EQUAL, Tracking.LOCAL]:
            foreground = ansi('redf')
        elif self._is_current:
            foreground = ansi('brightblackf')
        return foreground + self._local_branch

    def color_track(self) -> str:
        return {
           Tracking.EQUAL: ansi('brightblackf') + 'eq',
           Tracking.LOCAL: ansi('whitef') + 'local',
           Tracking.LOCAL_BEHIND_REMOTE: ansi('brightyellowf') + f'<({self._behind})',
           Tracking.LOCAL_AHEAD_OF_REMOTE: ansi('brightyellowf') + f'>({self._ahead})',
           Tracking.DIVERGED: ansi('brightyellowf') + f'<>({self._ahead}, {self._behind})',
           Tracking.ERR: ansi('redf') + 'ERR',
        }[self._tracking]

    def color_flags(self) -> str:
        foreground = ansi('redf') if self._is_dirty() else ansi('whitef')
        return foreground + self._flags

    def color_tag(self) -> str:
        foreground = ansi('brightmagentaf')
        return foreground + self._tag

    def color_hash(self) -> str:
        foreground = ansi('brightblackf')
        return foreground + self._hash

    def _get_update_status(self) -> None:
        if self._manifest_branch is None:
            return Status.UNTRACKED
        return Status.NOMINAL

    def _get_flags(self) -> str:
        flags = ''
        flags += '*' if self._are_unstaged_changes else ' '
        flags += '+' if self._are_staged_changes else ' '
        flags += '%' if self._are_untracked_files else ' '
        flags += '$' if self._are_stashes else ' '
        return flags

    def _is_dirty(self) -> bool:
        # Note: stashes do not count towards 'dirtiness'.
        return any([self._are_unstaged_changes, self._are_staged_changes,
                    self._are_untracked_files])

    @staticmethod
    def _is_valid_branch_name(branch_name) -> bool:
        acceptable_full = ("master", "develop", "azevtec", "outrider")
        if branch_name in acceptable_full:
            return True
        prefixes = ("bugfix/", "demo/", "feature/", "hotfix/", "int/", "pilot/", "release/")
        for prefix in prefixes:
            if branch_name.startswith(prefix):
                return True
        return False

    def _get_tracking_status(self) -> Tracking:
        if not self._remote_branch:
            return Tracking.LOCAL
        if self._ahead == 0 and self._behind == 0:
            return Tracking.EQUAL
        if self._ahead == 0 and self._behind > 0:
            return Tracking.LOCAL_BEHIND_REMOTE
        if self._ahead > 0 and self._behind == 0:
            return Tracking.LOCAL_AHEAD_OF_REMOTE
        if self._ahead > 0 and self._behind > 0:
            return Tracking.DIVERGED
        return Tracking.ERR


def get_manifest_branch(manifest_repos: Dict[str, Dict[str, Any]], path: str) -> str:
    if path not in manifest_repos:
        return None
    return manifest_repos[path].get("version", None)


def generate_table_entries(compares : Dict[str, Any], manifest_repos : Dict[str, Dict[str, Any]],
                           root_path: str, significant_only: bool) -> Dict[str, IEntry]:
    entries : Dict[str, IEntry] = {}
    existing_paths = set()
    # Add entries found on the filesystem (but may be mssing from the manifest).
    for compare in compares:
        # Strip the input path from the client path
        path = compare['cwd'].replace(f'{root_path}/', '')
        existing_paths.add(path)
        manifest_branch = get_manifest_branch(manifest_repos, path)
        entry = ExistingRepoTableEntry(path, compare['output'], manifest_branch)
        if (significant_only and not entry.is_significant()):
            continue
        entries[path] = entry
    # Add entries which exist in the manifest but are missing from the filesystem.
    for path in manifest_repos:
        if path not in existing_paths:
            entries[path] = MissingRepoTableEntry(path)
    return entries


def read_repos_from_manifest_file(manifest_file_path: str) -> Dict[str, Dict[str, Any]]:
    try:
        with open(manifest_file_path, 'r', encoding='utf-8') as manifest_file:
            return get_repositories(manifest_file)
    except RuntimeError as ex:
        print(ansi('redf') + str(ex) + ansi('reset'), file=sys.stderr)
    return None


def main(args=None, stdout=None, stderr=None) -> None:
    set_streams(stdout=stdout, stderr=stderr)

    parser = CompareCommand.get_parser()
    add_common_arguments(parser, skip_hide_empty=True, skip_nested=True, skip_repos=True,
                         path_nargs='?')
    args = parser.parse_args(args)

    manifest_repos = read_repos_from_manifest_file(args.input)
    if manifest_repos is None:
        return 1

    compare = CompareCommand(args).execute()
    entries = generate_table_entries(compare, manifest_repos, root_path=args.path,
                                     significant_only=args.significant)
    RepoTable(entries).print(manifest_file=args.input)
    return 0


if __name__ == '__main__':
    sys.exit(main())
