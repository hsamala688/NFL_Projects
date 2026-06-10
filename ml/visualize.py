# Inspired by information provided by Ngozi Harrison as part of Final Paper for Cluster 10CW
from pathlib import Path
import xgboost as xgb

ML_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(__file__).parent / "xgb_model.json"

booster = xgb.Booster()
booster.load_model(str(MODEL_PATH))

ML_DIR.mkdir(parents=True, exist_ok=True)
graph = xgb.to_graphviz(booster, tree_idx=0, rankdir="TB")
graph.render(filename="tree_plot", directory=str(ML_DIR), format="png", cleanup=True)

print("Wrote", ML_DIR / "tree_plot")
