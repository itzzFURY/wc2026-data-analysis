# WC2026 Data Analysis

Team project for the Foundation of Data Science assignment, analysing FIFA World Cup 2026 data.

## What's in here

**Part 1** - 4 analytic tasks, each covering 6 skills (question formulation, data wrangling, sampling, descriptive statistics, confidence interval, two-sample t-test).

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

## Everyday workflow

Everyone works directly on `main`. Pull before you start, push when you're done.

```
git pull
```

*(do your work, save your notebook)*

```
git add .
git commit -m "describe what you did"
git push
```

## Task assignments

| Task | Topic | Assigned to |
|---|---|---|
| Task 1 | What proportion of eligible outfield players committed at least one foul,
and did defenders and forwards differ significantly in mean fouls per 90 minutes at the 2026 World Cup?| Shreeya |
| Task 2 | | |
| Task 3 | | |
| Task 4 | Do teams with a faster average top speed score significantly 
more goals per match than teams with a slower average top speed in the 2026 World Cup? | Vibhi |

## Data sources

- FIFA official statistics: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics
- The Stats Don't Lie: https://www.thestatsdontlie.com/football/world-cup-2026/
- FBref: https://fbref.com/en/
