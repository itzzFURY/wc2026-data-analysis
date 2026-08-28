"""
SKILL 1: QUESTION FORMULATION
Analytic Question: Among outfield players who played at least 90 minutes at the
FIFA World Cup 2026, what proportion committed at least one foul, and did
defenders and forwards differ in their mean number of fouls committed per 90
minutes?

This file implements the early analysis steps only: loading, inspection,
cleaning, defining the eligible population, and creating derived features.
"""

import pandas as pd
import numpy as np
import re

# SKILL 2: DATA WRANGLING (Acquisition, Cleaning, & Feature Construction)

# 1. Load the raw FBref player dataset
# The raw CSV is left unchanged; all cleaning and checks happen in Python.
# 2. Inspect the raw data briefly so we know what we have before modifying it.

RAW_PATH = 'task_fouls/fifa2026_player_misc_raw.csv'  # relative to repo root

print('Loading raw dataset from:', RAW_PATH)
df_raw = pd.read_csv(RAW_PATH, encoding='utf-8')

print('\n-- RAW DATA INSPECTION --')
print('First 5 rows:')
print(df_raw.head(5))
print('\nShape:', df_raw.shape)
print('\nColumn names:')
print(list(df_raw.columns))
print('\nData types:')
print(df_raw.dtypes)
print('\nMissing values per column:')
print(df_raw.isna().sum())
print('\nDuplicate rows count: ', df_raw.duplicated().sum())


# 3. Select the variables needed for this analysis
# Keep only player name, position, team, 90s (playing time) and fouls.

# 4. Rename columns to simpler names for the rest of the script.
cols_keep = ['Player', 'Pos', 'Squad', '90s', 'Fls']
missing_cols = [c for c in cols_keep if c not in df_raw.columns]
if missing_cols:
    raise ValueError(f"Expected columns missing from raw file: {missing_cols}")

df = df_raw[cols_keep].copy()
df = df.rename(columns={
    'Player': 'player',
    'Pos': 'position',
    'Squad': 'team',
    '90s': 'nineties_played',
    'Fls': 'fouls_committed'
})

print('\n-- AFTER SELECT & RENAME --')
print(df.head(5))
print('\nDtypes before numeric conversion:')
print(df.dtypes)


# 5. Convert numerical columns
# Turn `nineties_played` and `fouls_committed` into numeric types. Any values
# that can't be parsed become NaN so we don't guess or invent data.
df['nineties_played'] = pd.to_numeric(df['nineties_played'], errors='coerce')
df['fouls_committed'] = pd.to_numeric(df['fouls_committed'], errors='coerce')

print('\nMissing values in selected columns (after numeric coercion):')
print(df[['player', 'position', 'team', 'nineties_played', 'fouls_committed']].isna().sum())


# 6. Clean the team names
# FBref sometimes stores teams like "us USA" or "no Norway". Remove short
# leading/trailing country codes while keeping the full country name.
def clean_team(name):
    if pd.isna(name):
        return name
    s = str(name).strip()
    # remove leading or trailing 2-3 letter country codes (like 'us ', ' mx', case-insensitive)
    s = re.sub(r'^[A-Za-z]{2,3}\s+', '', s)
    s = re.sub(r'\s+[A-Za-z]{2,3}$', '', s)
    return s.strip()

# work on a cleaned copy so the original selection remains available
df_clean = df.copy()
df_clean['team'] = df_clean['team'].apply(clean_team)

print('\nSample cleaned team values:')
print(df_clean['team'].dropna().unique()[:20])


# 7. Check for missing required fields
# We must identify rows missing `position`, `nineties_played`, or
# `fouls_committed` before we filter the population. We report examples below.
required = ['position', 'nineties_played', 'fouls_committed']
missing_required_mask = df_clean[required].isna().any(axis=1)
missing_required_count = missing_required_mask.sum()
print(f"\nRows with missing required fields (position, nineties_played, fouls_committed): {missing_required_count}")
if missing_required_count > 0:
    print('Examples of rows with missing required fields:')
    print(df_clean[missing_required_mask].head(5))


# 8. Define the eligible population
# Eligibility: outfield players (exclude exact 'GK') who have >= 1.0 in `90s`.
# We count players with >=1.0 90s before removing goalkeepers for transparency.
players_with_1plus_90s = df_clean[df_clean['nineties_played'] >= 1.0].shape[0]

# Remove rows with missing required fields (reported above)
df_eligible = df_clean.dropna(subset=required).copy()

# Exclude goalkeepers (exact match 'GK')
df_eligible = df_eligible[df_eligible['position'] != 'GK'].copy()

# Then require at least 1.0 nineties
df_eligible = df_eligible[df_eligible['nineties_played'] >= 1.0].copy()

print('\n-- POPULATION COUNTS & FREQUENCIES --')
print('Raw dataset rows:', len(df_raw))
print('Players with >= 1.0 90s (before excluding GKs):', players_with_1plus_90s)
print('Final eligible outfield population size:', len(df_eligible))
print('\nFrequency table of positions (eligible population):')
print(df_eligible['position'].value_counts(dropna=False))


# 9. Feature construction
# `committed_foul` flags whether a player committed at least one foul (1/0).
# `fouls_per_90` standardises fouls by playing time so players are comparable.
df_eligible['committed_foul'] = np.where(df_eligible['fouls_committed'] > 0, 1,
                                         np.where(df_eligible['fouls_committed'] == 0, 0, np.nan))

# fouls_per_90: fouls_committed divided by nineties_played
df_eligible['fouls_per_90'] = df_eligible['fouls_committed'] / df_eligible['nineties_played']


# 10. Validation and output
# Print the first 10 cleaned rows and a few counts so you can verify the
# filtering and new variables before any statistical analysis.
cols_out = ['player', 'position', 'team', 'nineties_played', 'fouls_committed', 'committed_foul', 'fouls_per_90']
print('\nFirst 10 rows of the cleaned eligible population:')
print(df_eligible[cols_out].head(10).to_string(index=False))

eligible_size = len(df_eligible)
num_committed = int(df_eligible['committed_foul'].sum()) if df_eligible['committed_foul'].notna().any() else 0
num_zero = int((df_eligible['committed_foul'] == 0).sum())
num_DF = int((df_eligible['position'] == 'DF').sum())
num_FW = int((df_eligible['position'] == 'FW').sum())

print('\nSummary validation:')
print('Eligible population size:', eligible_size)
print('Number who committed at least one foul:', num_committed)
print('Number who committed zero fouls:', num_zero)
print('Number of exact DF players:', num_DF)
print('Number of exact FW players:', num_FW)

if len(df_raw) != 1039 or eligible_size != 685:
    print('\nNOTE: expected raw rows=1039 and eligible=685. Values differ above;')
    print('Do not force counts. Check for differences in the raw file (missing rows,')
    print('column name mismatches, or unexpected data formats).')

print('\nScript complete: data loading, cleaning, population definition, and features are ready.')
