# WC2026 Data Analysis

Team project for the Foundation of Data Science assignment, analysing FIFA World Cup 2026 data.

## What's in here

**Part 1** - 4 analytic tasks, each covering 6 skills (question formulation, data wrangling, sampling, descriptive statistics, confidence interval, two-sample t-test).

**Part 2** - 2 linear regression models predicting match outcomes using pre-match data only.

## Repo structure

```
data/
  raw/          # untouched data pulled from source sites, don't edit these files
  processed/    # cleaned tables ready for analysis
notebooks/      # one notebook per task, see naming convention below
src/            # shared helper functions (scraping, cleaning) used across notebooks
reports/        # final write-ups and charts for submission
requirements.txt
```

Empty folders contain a `.gitkeep` placeholder file so Git tracks them. It's safe to leave it there once you add real files.

## Setup

### Windows (PowerShell)

```
git clone https://github.com/itzzFURY/wc2026-data-analysis.git
cd wc2026-data-analysis

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

If activation is blocked by a script execution policy error, run this once (it's a one-time permission fix):

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Mac/Linux

```
git clone https://github.com/itzzFURY/wc2026-data-analysis.git
cd wc2026-data-analysis

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

You'll know the virtual environment is active when you see `(venv)` at the start of your terminal prompt.

## Working on a task

Create a branch named after your task before making changes, don't commit straight to `main`:

```
git checkout -b feature/task1-forward-goals
```

When your task is ready:

```
git add .
git commit -m "add task1 forward goals analysis"
git push -u origin feature/task1-forward-goals
```

Then open a pull request on GitHub into `main` so someone else can glance at it before it's merged.

## Notebook naming convention

Name notebooks after the task they cover, for example:

```
notebooks/task1_forward_goals.ipynb
notebooks/task2_passing.ipynb
notebooks/task3_cards.ipynb
notebooks/task4_distance.ipynb
notebooks/regression_goal_diff.ipynb
notebooks/regression_team_goals.ipynb
```

## Task assignments

| Task | Topic | Assigned to |
|---|---|---|
| Task 1 | | |
| Task 2 | | |
| Task 3 | | |
| Task 4 | | |
| Regression 2.1 | Goal difference (104 rows, 8 variables) | |
| Regression 2.2 | Team goals scored (208 rows, 8 variables) | |

## Data sources

- FIFA official statistics: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics
- The Stats Don't Lie: https://www.thestatsdontlie.com/football/world-cup-2026/
- FBref: https://fbref.com/en/