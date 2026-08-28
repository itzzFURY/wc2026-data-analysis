import pandas as pd
import re

tables = pd.read_html('World Cup Scores & Fixtures _ FBref.com.html')
print("Tables found:", len(tables))

matches = tables[9].copy()
print(matches.shape)
print(matches.columns.tolist())

# Keep only real match rows: drop blank rows and any repeated header rows
matches = matches[matches['Home'].notna() & matches['Score'].notna()]
matches = matches[matches['Home'].astype(str).str.strip() != 'Home'].reset_index(drop=True)
print("After removing blanks/headers:", matches.shape)

# FBref writes teams like 'mx Mexico' or 'Mexico mx' - strip the country code
def clean_team(name):
    name = str(name).strip()
    name = re.sub(r'^[a-z]{2,3}\s+', '', name)   # remove code at the start
    name = re.sub(r'\s+[a-z]{2,3}$', '', name)   # remove code at the end
    return name.strip()

matches['Home'] = matches['Home'].apply(clean_team)
matches['Away'] = matches['Away'].apply(clean_team)
print(matches[['Round', 'Date', 'Home', 'Score', 'Away']].head(8))

# Split score into numbers (handles shootouts like '(3) 1–1 (4)' too)
def parse_score(score):
    score = str(score).replace('–', '-')
    m = re.search(r'(\d+)\s*-\s*(\d+)', score)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

matches[['home_goals', 'away_goals']] = matches['Score'].apply(lambda s: pd.Series(parse_score(s)))

# Remove only 0-0 matches (no first goal exists)
matches = matches[~((matches['home_goals'] == 0) & (matches['away_goals'] == 0))]

# Save to CSV
matches.to_csv('task2_matches.csv', index=False, encoding='utf-8-sig')
print("Saved", len(matches), "matches")