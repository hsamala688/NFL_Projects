Current Project Structure:
NFL-Projects/
├── data/
│  team-logs/         ← all the team offensive/run play CSVs
│   player-logs/      ← C.Sutton, JSN, M.Mims, T.Franklin CSVs
│   play-by-play/     ← play_by_play_2025.csv
├── Bo-Nix-Analysis/
│   ├── data/        ← Bo Nix CSVs
│   ├── notebooks/   ← all Bo Nix notebooks (consolidating subfolders)
├── notebooks/       ← Comprehensive_Player_Analysis and future notebooks
├── .gitignore
├── README.md
└── requirements.txt

Future Planned Structure:
NFL-Projects/
├── data/
│   ├── raw/                # Original nflreadpy Parquet files (Season-by-Season)
│   └── processed/          # Filtered data (e.g., "all_rookie_qbs.parquet")
├── src/                    # THE ENGINE (The most important part for a portfolio)
│   ├── __init__.py
│   ├── loader.py           # nflreadpy logic
│   ├── analytics.py        # Polars-based math (EPA, CPOE, Accuracy)
│   └── visuals.py          # Standardized plotting functions (Plotly/Matplotlib)
├── notebooks/
│   ├── bo_nix_deep_dive.ipynb
│   ├── league_wide_rookie_trends.ipynb
│   └── team_offense_reports.ipynb
├── .gitignore              # Ignore data/raw/*.parquet
├── README.md
└── requirements.txt
