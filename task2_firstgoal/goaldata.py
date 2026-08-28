import time
import pandas as pd
import requests

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"

def fetch_all_tournament_goals():
    print("Fetching full World Cup tournament schedule from ESPN...")
    
    # Pass full tournament date range (June 11 to July 19, 2026) and limit
    scoreboard_params = {
        "dates": "20260611-20260719",
        "limit": 500
    }
    
    scoreboard_res = requests.get(f"{BASE_URL}/scoreboard", params=scoreboard_params)
    data = scoreboard_res.json()
    
    events = data.get("events", [])
    print(f"Retrieved {len(events)} tournament matches. Fetching goal timings...\n")
    
    all_goals = []

    for idx, event in enumerate(events, 1):
        game_id = event["id"]
        match_name = event["name"]
        match_date = event["date"][:10]
        status = event.get("status", {}).get("type", {}).get("shortDetail", "-")
        
        # Query individual match summary for key events
        summary_res = requests.get(f"{BASE_URL}/summary", params={"event": game_id})
        summary_data = summary_res.json()
        
        # Soccer events in ESPN are stored in 'keyEvents' or 'scoringPlays'
        raw_events = summary_data.get("keyEvents", []) or summary_data.get("scoringPlays", [])
        goals = [e for e in raw_events if "goal" in e.get("type", {}).get("text", "").lower()]
        
        if not goals:
            all_goals.append({
                "Date": match_date,
                "Match": match_name,
                "Status": status,
                "Minute": "-",
                "Scorer": "No Goals / Scoreless Draw",
                "Team": "-",
                "Type": "-"
            })
        else:
            for g in goals:
                minute = g.get("clock", {}).get("displayValue", "-")
                
                # Extract player name
                athletes = g.get("athletesInvolved", [])
                scorer = athletes[0].get("displayName", "Unknown") if athletes else "Unknown"
                
                # Extract team and goal type
                team = g.get("team", {}).get("displayName", "-")
                goal_type = g.get("type", {}).get("text", "Goal")
                
                all_goals.append({
                    "Date": match_date,
                    "Match": match_name,
                    "Status": status,
                    "Minute": minute,
                    "Scorer": scorer,
                    "Team": team,
                    "Type": goal_type
                })
        
        time.sleep(0.3)  # Brief pause to prevent rate limiting

    df = pd.DataFrame(all_goals)
    df.to_csv("all_world_cup_goals_espn.csv", index=False)
    print(f"\nDone! Extracted {len(df)} goal/match records to 'all_world_cup_goals_espn.csv'.")
    return df

# Run the extraction
df_all = fetch_all_tournament_goals()
df_all.head(25)