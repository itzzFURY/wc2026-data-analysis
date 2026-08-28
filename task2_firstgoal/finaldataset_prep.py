import pandas as pd

# read the two source files
matches = pd.read_csv("task2_matches.csv")
goals = pd.read_csv("all_world_cup_goals_espn.csv")

# drop 0-0 games (no first goal)
matches = matches[~((matches["home_goals"] == 0) & (matches["away_goals"] == 0))].reset_index(drop=True)

# turn goal minute into a number, handle stoppage time like "45+2"
def get_min(m):
    m = str(m).replace("'", "").strip()
    if m in ("-", ""):
        return None
    if "+" in m:
        return int(m.split("+")[0])
    try:
        return int(m)
    except:
        return None

def sort_min(m):
    m = str(m).replace("'", "").strip()
    if "+" in m:
        a, b = m.split("+")
        return int(a) + int(b) / 100
    try:
        return float(m)
    except:
        return 999

goals["base_min"] = goals["Minute"].apply(get_min)
goals["sort_min"] = goals["Minute"].apply(sort_min)
goals = goals.dropna(subset=["base_min"])

# get the first goal of each match (smallest minute)
first = goals.sort_values("sort_min").groupby("Match", as_index=False).first()
first[["espn_away", "espn_home"]] = first["Match"].str.split(" at ", expand=True)

# match team names between the two files
name_map = {
    "United States": "USA", "South Korea": "Korea Republic",
    "Bosnia-Herzegovina": "Bosnia–Herz", "DR Congo": "Congo DR",
    "Cape Verde": "Cabo Verde", "Iran": "IR Iran",
    "Cote d'Ivoire": "Côte d'Ivoire", "Ivory Coast": "Côte d'Ivoire",
    "Turkey": "Türkiye", "Turkiye": "Türkiye",
}
def fix_name(x):
    return name_map.get(str(x).strip(), str(x).strip())

first["home_std"] = first["espn_home"].apply(fix_name)
first["away_std"] = first["espn_away"].apply(fix_name)
matches["home_std"] = matches["Home"].astype(str).str.strip()
matches["away_std"] = matches["Away"].astype(str).str.strip()

# merge first goal info into the matches
merged = matches.merge(first[["home_std", "away_std", "Minute", "base_min", "Team", "Type"]],
                       on=["home_std", "away_std"], how="left")

# France-Paraguay had missing goal data in the source, fill it manually (Mbappe pen 70')
fix = (merged["Home"] == "Paraguay") & (merged["Away"] == "France")
merged.loc[fix, "Minute"] = "70'"
merged.loc[fix, "base_min"] = 70
merged.loc[fix, "Team"] = "France"
merged.loc[fix, "Type"] = "Goal - Penalty"

# drop the one first goal in extra time (match was 0-0 at 90 min)
merged = merged[merged["base_min"] <= 90].reset_index(drop=True)

# rename and build the columns we need
merged = merged.rename(columns={"Minute": "first_goal_minute", "base_min": "first_goal_min",
                                "Team": "first_scorer_team", "Type": "goal_type"})

def winner(r):
    if r["home_goals"] > r["away_goals"]:
        return r["Home"]
    if r["away_goals"] > r["home_goals"]:
        return r["Away"]
    return "Draw"

merged["winner"] = merged.apply(winner, axis=1)
merged["won"] = (merged["first_scorer_team"] == merged["winner"]).astype(int)

med = merged["first_goal_min"].median()
merged["half"] = merged["first_goal_min"].apply(lambda x: "1st" if x <= 45 else "2nd")
merged["timing_group"] = merged["first_goal_min"].apply(lambda x: "early" if x <= med else "late")

# keep only the columns we need
final = merged[["Round", "Date", "Home", "Away", "home_goals", "away_goals",
                "first_scorer_team", "first_goal_minute", "first_goal_min", "goal_type",
                "winner", "won", "half", "timing_group"]].copy()
final.insert(0, "match_id", range(1, len(final) + 1))

final.to_csv("task2_final_dataset.csv", index=False, encoding="utf-8-sig")
print("Saved", len(final), "matches to task2_final_dataset.csv")