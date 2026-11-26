"""Import every module in the repo, so a missing pin fails CI instead of a run.

This exists because requirements.txt drifted out of sync with the code's real
imports: the file listed 13 packages while the project imported more than that,
and nothing caught it until someone built a clean environment by hand.

Run from the repo root:  python ci/check_imports.py
"""
import importlib
import os
import py_compile
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# matplotlib picks an interactive backend by default and there is no display
# on a CI runner, so ml.train's `import matplotlib.pyplot` would fail.
os.environ.setdefault("MPLBACKEND", "Agg")

# ── modules that are safe to import ──────────────────────────────────────────
# Every module whose import does nothing but define names. Database connections
# in this repo all happen inside functions, so these import fine on a runner
# with no db/nfl.duckdb and no data/raw/ (both gitignored).
IMPORT_MODULES = [
    "src.analytics",
    "src.loader",
    "src.visuals",
    "ingestion.pull_pbp",
    "ingestion.load_duckdb",
    "serving.build_serving_db",
    "ml.train",
    "ml.predict",
    "agent.semantic_layer",
    "agent.tools",
    "agent.agent",
    "agent.respond",
    "backend.config",
    "backend.cache",
    "backend.schemas",
    "backend.main",
    "flows.nfl_pipeline",
]

# ── modules that can only be syntax checked ──────────────────────────────────
# dashboard/app.py is a Streamlit entrypoint: it calls st.* at module level and
# is meant to be executed by `streamlit run`, not imported. Importing it here
# would run its UI construction outside a Streamlit session. Compiling it still
# catches syntax errors and leaves the import sweep honest about what it covers.
COMPILE_ONLY = [
    "dashboard/app.py",
    "ml/visualize.py",
]

# ── modules with import-time side effects ────────────────────────────────────
# ml/visualize.py is compile-only rather than imported, deliberately. It does
# real work at import time (ml/visualize.py:8-13): it loads ml/xgb_model.json
# into a Booster, calls xgb.to_graphviz(), and renders ml/tree_plot.png by
# shelling out to graphviz's `dot`. So importing it here would fail for a reason
# unrelated to our pins, because `dot` is not reliably installed on
# ubuntu-latest, and it rewrites a tracked file on every run.
#
# Almost no dependency coverage is lost. It is a leaf module that nothing else
# in the repo imports, and its only non-stdlib import is xgboost, which four
# modules in IMPORT_MODULES already exercise. Doing I/O at import time is the
# actual defect, and papering over it in CI would only hide that. Move it into
# IMPORT_MODULES once the rendering sits behind a main() guard.
SIDE_EFFECT_MODULES: list[str] = []


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    failures: list[tuple[str, str]] = []

    to_import = IMPORT_MODULES + SIDE_EFFECT_MODULES
    print(f"importing {len(to_import)} modules")
    for name in to_import:
        try:
            importlib.import_module(name)
            print(f"  ok   {name}")
        except Exception as e:
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
            failures.append((name, f"{type(e).__name__}: {e}"))

    print(f"\ncompiling {len(COMPILE_ONLY)} files")
    for rel in COMPILE_ONLY:
        path = REPO_ROOT / rel
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"  ok   {rel}")
        except Exception as e:
            print(f"  FAIL {rel}: {type(e).__name__}: {e}")
            failures.append((rel, f"{type(e).__name__}: {e}"))

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for name, err in failures:
            print(f"  {name}: {err}")
        print(
            "\nA module failing to import usually means requirements.txt is "
            "missing a pin for something the code imports."
        )
        return 1

    print(f"\nall {len(to_import)} imports and {len(COMPILE_ONLY)} compiles passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
