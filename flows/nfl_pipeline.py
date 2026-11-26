"""
The weekly pipeline, end to end.

    pull -> load views -> dbt run -> dbt test -> build serving db -> publish

The last two stages are the point of the flow existing. Before they were
wired in, the flow stopped at `dbt test` and the serving database the agent
actually reads was built by hand, which is why weekly automation did not
work. Scheduling was never the missing piece.

Invoked as a one shot command, not a daemon. A weekly job should not need a
process alive for seven days to learn what time it is, so the schedule lives
in Cloud Scheduler and this exits when the work is done.

    python flows/nfl_pipeline.py --seasons 2025
    python flows/nfl_pipeline.py --seasons 2025 --out-dir build/artifacts
    python flows/nfl_pipeline.py --seasons 2025 --bucket my-bucket
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from prefect import flow, task

# Resolve the repo root relative to this file so paths work
# regardless of where you invoke the script from
REPO_ROOT = Path(__file__).parent.parent
DBT_PROJECT_DIR = REPO_ROOT / "nfl_analytics"


def _run(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


@task(name="Pull PBP Data")
def pull_pbp_task(seasons: list[int]):
    """Download play-by-play Parquet files for the given seasons."""
    _run(
        [sys.executable, "ingestion/pull_pbp.py", "--seasons", *[str(s) for s in seasons]],
        REPO_ROOT,
    )


@task(name="Load DuckDB Views")
def load_duckdb_task():
    """Register Parquet files as DuckDB views."""
    _run([sys.executable, "ingestion/load_duckdb.py"], REPO_ROOT)


@task(name="dbt Run")
def dbt_run_task():
    """Run all dbt models."""
    _run(["dbt", "run"], DBT_PROJECT_DIR)


@task(name="dbt Test")
def dbt_test_task():
    """Run all dbt tests."""
    _run(["dbt", "test"], DBT_PROJECT_DIR)


@task(name="Build Serving DB")
def build_serving_db_task():
    """
    Materialize the marts into db/serving.duckdb as real tables.

    The agent connects with enable_external_access=False, so the parquet
    backed views in db/nfl.duckdb are unreadable to it and real tables are
    required. This is the stage that used to be run by hand.
    """
    _run([sys.executable, "serving/build_serving_db.py"], REPO_ROOT)


@task(name="Publish Artifacts")
def publish_task(bucket: str | None, out_dir: str | None):
    """
    Write stats.json and then latest/manifest.json, pointer last.

    Skipped when there is nowhere to publish, so a local run of the flow
    still exercises every stage above it.
    """
    if not bucket and not out_dir:
        print("no --bucket and no --out-dir: nothing to publish")
        return
    cmd = [sys.executable, "serving/publish.py"]
    if bucket:
        cmd += ["--bucket", bucket]
    if out_dir:
        cmd += ["--out-dir", out_dir]
    _run(cmd, REPO_ROOT)


@flow(name="NFL Pipeline")
def nfl_pipeline(
    seasons: list[int] = [2025],
    bucket: str | None = None,
    out_dir: str | None = None,
):
    pull_pbp_task(seasons)
    load_duckdb_task()
    dbt_run_task()
    dbt_test_task()
    build_serving_db_task()
    publish_task(bucket, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2025],
        help="Seasons to pull. Defaults to 2025.",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("ARTIFACT_BUCKET"),
        help="Cloud Storage bucket to publish into. Defaults to $ARTIFACT_BUCKET.",
    )
    parser.add_argument(
        "--out-dir",
        help="Also write the artifacts to this local directory.",
    )
    args = parser.parse_args()
    nfl_pipeline(seasons=args.seasons, bucket=args.bucket, out_dir=args.out_dir)
