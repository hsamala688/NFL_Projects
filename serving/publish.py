"""
Turn a finished pipeline run into the two artifacts the portfolio edge reads.

Contract, from HaydenPortfolio PLAN.md section 2:

    gs://<bucket>/runs/<season>-w<week>/stats.json   the payload
    gs://<bucket>/latest/manifest.json               the pointer, written LAST

The pointer goes last because Cloud Storage is atomic per object but not
across objects. A fresh stats.json sitting beside last week's pointer is
simply invisible, which is a safe state. A fresh pointer beside a half
written payload is a broken read. Writing the pointer last is what makes a
whole run atomic from the consumer's side.

The consumer validates more than it parses. Fields it will reject a run over:
schema_version must equal 1, artifact_path must be non-empty with no "." or
".." segments, season, completed_week and upcoming_week must all be non-zero,
and generated_at must be RFC 3339. row_counts is carried for humans; the edge
does not read it. Everything below the manifest, qb_cpoe and
fantasy_projections both, is opaque bytes to the edge, which is what lets a
column be added here without a coordinated deploy.

Run from the repo root:
    python serving/publish.py --out-dir build/artifacts
    python serving/publish.py --bucket my-bucket
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import xgboost as xgb
from google.cloud import storage

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.predict import (  # noqa: E402
    CPOE_COLUMNS,
    MODEL_PATH,
    compute_qb_cpoe,
    prepare_features,
)

SERVING_DB = REPO_ROOT / "db" / "serving.duckdb"

# Must equal `schemaVersion` in HaydenPortfolio server/main.go. The edge
# refuses any other value rather than rendering fields it cannot interpret,
# so bumping this without bumping that one degrades to the last good payload.
SCHEMA_VERSION = 1

# Reserved by the contract, filled in Phase 4. The edge never parses this key,
# so shipping the empty shape now costs nothing and means a consumer can code
# against the real structure before a model exists.
POSITIONS = ("QB", "RB", "WR", "TE")

# Counted into the manifest for humans reading a run after the fact.
# mart_player_week arrives in Phase 3 and should join this list then.
COUNTED_RELATIONS = ("mart_qb_season", "int_qb_plays")

# The regular season has been 18 weeks since 2021, 17 before that. int_qb_plays
# filters to season_type = 'REG', so no week above this ever appears in the
# data and no week above this can be projected.
REGULAR_SEASON_WEEKS = 18


def git_sha() -> str:
    """Short SHA of the commit that produced this run, or "unknown"."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        # A container built without .git is a normal way to run this, and it
        # is not worth failing a whole run over a provenance string.
        return "unknown"


def week_context(con: duckdb.DuckDBPyConnection) -> tuple[int, int, int]:
    """
    Derive (season, completed_week, upcoming_week) from the loaded data.

    completed_week is the newest week present. On the Tuesday run that is the
    week which just finished, since the flow pulls after Monday night.

    upcoming_week is the next week, clamped to the end of the regular season.
    Once week 18 has been played there is no week 19 to project, and the edge
    would not catch it: its only rule is that the field is non-zero, so a 19
    would sail through and render as "projections for week 19". The clamp
    therefore has to happen on this side.

    Through the offseason the clamp holds upcoming_week equal to
    completed_week, which reads as "week 18 is both the last completed and the
    last projectable week". Both stay non-zero, which is what the edge
    requires.

    The int() casts are deliberate. DuckDB hands back its own integer types,
    and json.dumps refuses anything that is not a Python int, so a run would
    fail at serialization rather than here.
    """
    season, completed_week = con.execute(
        """
        SELECT season, max(week)
        FROM int_qb_plays
        WHERE season = (SELECT max(season) FROM int_qb_plays)
        GROUP BY season
        """
    ).fetchone()
    upcoming_week = min(int(completed_week) + 1, REGULAR_SEASON_WEEKS)
    return int(season), int(completed_week), upcoming_week


def row_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    counts = {}
    for rel in COUNTED_RELATIONS:
        counts[rel] = con.execute(f'SELECT count(*) FROM "{rel}"').fetchone()[0]
    return counts


def build_manifest(
    season: int,
    completed_week: int,
    upcoming_week: int,
    counts: dict[str, int],
    generated_at: str | None = None,
) -> dict:
    """
    The pointer object, and the copy embedded in the payload.

    artifact_path is zero padded to two digits so runs sort lexically, which
    is what makes `gsutil ls runs/` readable in week order.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": season,
        "completed_week": completed_week,
        "upcoming_week": upcoming_week,
        "artifact_path": f"runs/{season}-w{completed_week:02d}/",
        "git_sha": git_sha(),
        "row_counts": counts,
    }


def qb_cpoe_rows(con: duckdb.DuckDBPyConnection, season: int) -> list[dict]:
    """Score one season's attempts and rank them, as CPOE_COLUMNS defines."""
    df = con.execute(
        """
        SELECT passer_player_id, passer_player_name, possession_team, season,
               week, down, yards_to_go, yards_from_endzone, air_yards,
               pass_location, score_differential, game_seconds_remaining,
               shotgun, no_huddle, is_complete
        FROM int_qb_plays
        WHERE season = ?
        """,
        [season],
    ).df()

    df, feature_cols = prepare_features(df)
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    # float64 because predict_proba returns float32, and float32 63.6 is
    # 63.599998474121094 once it reaches JSON.
    df["completion_prob"] = model.predict_proba(df[feature_cols])[:, 1].astype("float64")

    out = compute_qb_cpoe(df)
    assert list(out.columns) == CPOE_COLUMNS, f"CPOE shape drifted: {list(out.columns)}"
    return out.to_dict(orient="records")


def build_payload(
    con: duckdb.DuckDBPyConnection,
    season: int,
    completed_week: int,
    upcoming_week: int,
    generated_at: str | None = None,
) -> dict:
    """
    The whole stats.json, in the frozen key order.

    The manifest is embedded as well as written separately, so an archived
    runs/<...>/stats.json stays self describing once latest/manifest.json has
    moved on. The payload deliberately carries no "source" field: that is the
    edge's own record of where the bytes came from, and a payload must not be
    able to claim its own provenance.
    """
    manifest = build_manifest(
        season, completed_week, upcoming_week, row_counts(con), generated_at
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": manifest,
        "qb_cpoe": qb_cpoe_rows(con, season),
        "fantasy_projections": {pos: [] for pos in POSITIONS},
    }


def write_local(payload: dict, out_dir: Path) -> tuple[Path, Path]:
    """
    Write the same two objects to a local directory, same relative layout.

    This is how a run is inspected without a bucket, and how the shape is
    checked against the consumer's fixture.
    """
    manifest = payload["manifest"]
    run_dir = out_dir / manifest["artifact_path"].strip("/")
    run_dir.mkdir(parents=True, exist_ok=True)

    stats_path = run_dir / "stats.json"
    stats_path.write_text(json.dumps(payload, indent=2) + "\n")

    pointer_dir = out_dir / "latest"
    pointer_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = pointer_dir / "manifest.json"
    pointer_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return stats_path, pointer_path


def publish_to_gcs(bucket_name: str, payload: dict) -> tuple[str, str]:
    """
    Upload the payload, then the pointer. Order is the whole point.

    If this process dies between the two uploads, the bucket holds a complete
    new payload that nothing points at yet, and the edge keeps serving last
    week's run. That is the failure mode worth having.
    """
    manifest = payload["manifest"]
    bucket = storage.Client().bucket(bucket_name)

    stats_object = manifest["artifact_path"].strip("/") + "/stats.json"
    blob = bucket.blob(stats_object)
    blob.cache_control = "no-store"
    blob.upload_from_string(
        json.dumps(payload, indent=2) + "\n", content_type="application/json"
    )

    # Only now the pointer, and never cached: the edge polls it every 5 minutes
    # and a cached copy would hide a fresh run for as long as the TTL.
    pointer = bucket.blob("latest/manifest.json")
    pointer.cache_control = "no-store"
    pointer.upload_from_string(
        json.dumps(manifest, indent=2) + "\n", content_type="application/json"
    )

    return f"gs://{bucket_name}/{stats_object}", f"gs://{bucket_name}/latest/manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--bucket",
        default=os.environ.get("ARTIFACT_BUCKET"),
        help="Cloud Storage bucket to publish into. Defaults to $ARTIFACT_BUCKET, "
        "which is the same variable the edge reads, so one name configures both "
        "sides. Omit both to do nothing.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Also write the two objects here, in the same relative layout.",
    )
    parser.add_argument(
        "--serving-db",
        type=Path,
        default=SERVING_DB,
        help="DuckDB file to read. Defaults to db/serving.duckdb.",
    )
    args = parser.parse_args()

    if not args.bucket and not args.out_dir:
        parser.error("nothing to do: pass --bucket, --out-dir, or both")
    if not args.serving_db.exists():
        parser.error(f"{args.serving_db} does not exist: run build_serving_db first")

    # The agent reads this file with external access disabled, so read it the
    # same way here. A view that still needs a parquet path would fail now
    # rather than at request time.
    con = duckdb.connect(
        str(args.serving_db), read_only=True, config={"enable_external_access": False}
    )
    try:
        season, completed_week, upcoming_week = week_context(con)
        print(
            f"  season {season}, completed week {completed_week}, "
            f"upcoming week {upcoming_week}"
        )
        payload = build_payload(con, season, completed_week, upcoming_week)
    finally:
        con.close()

    print(f"  qb_cpoe rows: {len(payload['qb_cpoe']):,}")

    if args.out_dir:
        stats_path, pointer_path = write_local(payload, args.out_dir)
        print(f"  wrote {stats_path}")
        print(f"  wrote {pointer_path}  (pointer, last)")

    if args.bucket:
        stats_uri, pointer_uri = publish_to_gcs(args.bucket, payload)
        print(f"  uploaded {stats_uri}")
        print(f"  uploaded {pointer_uri}  (pointer, last)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
