"""
Materialize the marts from db/nfl.duckdb into db/serving.duckdb as real
tables. The agent connects to serving.duckdb with external access disabled,
so materialized tables (not parquet-backed views) are required.

Run from repo root:
    python serving/build_serving_db.py
"""
import duckdb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DB = REPO_ROOT / "db" / "nfl.duckdb"
SERVING_DB = REPO_ROOT / "db" / "serving.duckdb"
VIEWS = ["mart_qb_season", "int_qb_plays"]


def _sql_literal(value) -> str:
    """Single-quote a value for SQL, doubling any quote it contains."""
    return "'" + str(value).replace("'", "''") + "'"


def _remove_db(path: Path) -> None:
    """Delete a DuckDB file and the write-ahead log it leaves beside itself."""
    path.unlink(missing_ok=True)
    path.with_name(path.name + ".wal").unlink(missing_ok=True)


def build(
    src_db: Path = SRC_DB,
    serving_db: Path = SERVING_DB,
    views: list[str] = VIEWS,
) -> Path:
    """
    Copy each view into serving_db as a real table, DuckDB to DuckDB.

    The copy goes through ATTACH rather than a pandas frame, so DuckDB's own
    types land in the serving database instead of whatever pandas infers on
    the way through, and int_qb_plays never has to fit in memory at once.
    """
    serving_db.parent.mkdir(parents=True, exist_ok=True)

    # Build beside the real artifact and rename at the very end. Unlinking
    # first would destroy a working serving.duckdb before its replacement
    # exists, so any failure in between would leave nothing to serve.
    tmp_db = serving_db.with_name(serving_db.name + ".tmp")
    _remove_db(tmp_db)

    # dbt bakes its own catalog name into every view body, so int_qb_plays
    # reads "FROM nfl.main.stg_pbp". The attach alias is therefore not free:
    # it has to be the source file's stem, which is what DuckDB would pick on
    # its own and what dbt's profile named the database.
    catalog = src_db.stem

    con = duckdb.connect(str(tmp_db))
    try:
        con.execute(f"ATTACH {_sql_literal(src_db)} AS \"{catalog}\" (READ_ONLY)")
        for view in views:
            print(f"  materializing {view}...")
            con.execute(
                f'CREATE TABLE "{view}" AS '
                f'SELECT * FROM "{catalog}".main."{view}"'
            )
            count = con.execute(f'SELECT count(*) FROM "{view}"').fetchone()[0]
            print(f"  {view}: {count:,} rows")
        con.execute(f'DETACH "{catalog}"')
    except BaseException:
        # A half-built database is worse than none. Clear it out so the
        # failure leaves exactly the previous artifact and nothing else.
        con.close()
        _remove_db(tmp_db)
        raise
    con.close()

    tmp_db.replace(serving_db)
    print(f"serving.duckdb written to {serving_db}")
    return serving_db


if __name__ == "__main__":
    build()
