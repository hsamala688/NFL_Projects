"""
Materialize mart_qb_season and int_qb_plays from db/nfl.duckdb into
db/serving.duckdb as real tables. The agent connects to serving.duckdb
with external access disabled, so materialized tables (not parquet-backed
views) are required.

Run from repo root:
    python serving/build_serving_db.py
"""
import duckdb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DB = REPO_ROOT / "db" / "nfl.duckdb"
SERVING_DB = REPO_ROOT / "db" / "serving.duckdb"
VIEWS = ["mart_qb_season", "int_qb_plays"]


def build():
    SERVING_DB.parent.mkdir(parents=True, exist_ok=True)
    SERVING_DB.unlink(missing_ok=True)

    src = duckdb.connect(str(SRC_DB), read_only=True)
    dst = duckdb.connect(str(SERVING_DB))

    for view in VIEWS:
        print(f"  materializing {view}...")
        df = src.execute(f"SELECT * FROM {view}").df()
        dst.execute(f"CREATE TABLE {view} AS SELECT * FROM df")
        count = dst.execute(f"SELECT count(*) FROM {view}").fetchone()[0]
        print(f"  {view}: {count:,} rows")

    src.close()
    dst.close()
    print(f"serving.duckdb written to {SERVING_DB}")


if __name__ == "__main__":
    build()
