"""Register raw Parquet files as views in DuckDB."""
import duckdb
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "nfl.duckdb"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

def load(db_path=DB_PATH, raw_dir=RAW_DIR):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))

    # Register each season as its own view
    for f in sorted(raw_dir.glob("pbp_*.parquet")):
        season = f.stem.split("_")[1]
        conn.execute(f"""
            CREATE OR REPLACE VIEW pbp_{season} AS
            SELECT * FROM read_parquet('{f}')
        """)
        print(f"  → registered view: pbp_{season}")

    # Unified view across all seasons (glob pattern)
    conn.execute(f"""
        CREATE OR REPLACE VIEW pbp AS
        SELECT * FROM read_parquet('{raw_dir}/pbp_*.parquet')
    """)
    print("  → registered unified view: pbp")
    conn.close()

if __name__ == "__main__":
    load()