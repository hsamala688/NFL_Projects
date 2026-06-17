import subprocess
import sys
from pathlib import Path

from prefect import flow, task

# Resolve the repo root relative to this file so paths work
# regardless of where you invoke the script from
REPO_ROOT = Path(__file__).parent.parent
DBT_PROJECT_DIR = REPO_ROOT / "nfl_analytics"


@task(name="Pull PBP Data")
def pull_pbp_task(seasons: list[int]):
    """Download play-by-play Parquet files for the given seasons."""
    season_args = [str(s) for s in seasons]
    result = subprocess.run(
        [sys.executable, "ingestion/pull_pbp.py", "--seasons", *season_args],
        cwd=REPO_ROOT,
        check=True,
    )


@task(name="Load DuckDB Views")
def load_duckdb_task():
    """Register Parquet files as DuckDB views."""
    subprocess.run(
        [sys.executable, "ingestion/load_duckdb.py"],
        cwd=REPO_ROOT,
        check=True,
    )


@task(name="dbt Run")
def dbt_run_task():
    """Run all dbt models."""
    subprocess.run(
        ["dbt", "run"],
        cwd=DBT_PROJECT_DIR,
        check=True,
    )


@task(name="dbt Test")
def dbt_test_task():
    """Run all dbt tests."""
    subprocess.run(
        ["dbt", "test"],
        cwd=DBT_PROJECT_DIR,
        check=True,
    )


@flow(name="NFL Pipeline")
def nfl_pipeline(seasons: list[int] = [2025]):
    pull_pbp_task(seasons)
    load_duckdb_task()
    dbt_run_task()
    dbt_test_task()


if __name__ == "__main__":
    import sys

    if "--run-now" in sys.argv:
        # Manual one-shot trigger
        nfl_pipeline()
    else:
        # Start the scheduler to keep the process alive, waiting for cron
        nfl_pipeline.serve(
            name="nfl-pipeline-scheduled",
            cron="0 6 * * 2",  # Every Tuesday at 6:00 AM
        )