from pathlib import Path
import argparse
import nflreadpy as nfl

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

def pull_pbp(seasons: list[int]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        df = nfl.load_pbp(seasons=[season])  # returns a Polars DataFrame
        out_path = RAW_DIR / f"pbp_{season}.parquet"
        df.write_parquet(out_path, compression="snappy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=[2025])
    args = parser.parse_args()
    pull_pbp(args.seasons)