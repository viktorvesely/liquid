#!/usr/bin/env python3

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


FOLDER_RE = re.compile(
    r"^(?P<experiment>exp_.+?)_"
    r"(?:(?P<launch_id>[0-9a-f]{10})_)?"
    r"(?P<task>[A-Za-z0-9]+)_"
    r"(?P<date>\d{8})_"
    r"(?P<time>\d{6})$"
)

RUN_RE = re.compile(r"run_(\d{5})")


def find_runs(folder: Path) -> set[int]:
    runs = set()

    for path in folder.rglob("*"):
        match = RUN_RE.search(path.name)
        if match:
            runs.add(int(match.group(1)))

    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path, nargs="?", default=(Path(__file__).parent / "runs"))
    args = parser.parse_args()

    runs_dir = args.runs_dir

    if not runs_dir.is_dir():
        raise SystemExit(f"Not a directory: {runs_dir}")

    # experiment -> list[(timestamp, launch_id, task, runs)]
    groups = defaultdict(list)

    for folder in runs_dir.iterdir():
        if not folder.is_dir():
            continue

        match = FOLDER_RE.match(folder.name)
        if match is None:
            continue

        timestamp = datetime.strptime(
            f"{match['date']}_{match['time']}",
            "%Y%m%d_%H%M%S",
        )

        experiment = match["experiment"]
        launch_id = match["launch_id"] or "<legacy>"
        task = match["task"]

        groups[experiment].append(
            (
                timestamp,
                launch_id,
                task,
                find_runs(folder),
                folder
            )
        )

    for experiment in sorted(groups):
        print(experiment)

        entries = sorted(
            groups[experiment],
            key=lambda x: x[0],
        )

        for i, (timestamp, launch_id, task, runs, folder) in enumerate(entries):
            is_last = i == len(entries) - 1
            branch = "└──" if is_last else "├──"

            count = len(runs)
            reached = max(runs, default=0)

            if not runs:
                continue
            else:
                missing = set(range(1, reached + 1)) - runs

                if missing:
                    status = (
                        f"{count} / {reached}, "
                        f"{len(missing)} missing"
                    )
                else:
                    status = f"{count} / {reached}"

            print(
                f"{branch} "
                f"{timestamp:%Y-%m-%d %H:%M:%S}  "
                f"{launch_id}  "
                f"{task:<10}  "
                f"{status}        "
                f"{folder.name}"
            )


if __name__ == "__main__":
    main()