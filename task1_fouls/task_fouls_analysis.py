"""
SKILL 1: ANALYSIS
Question I asked and tried to answer for this task: What proportion of eligible outfield players committed at least one foul,
and did defenders and forwards differ significantly in mean fouls per 90 minutes at the 2026 World Cup?

"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import stats

# SKILL 2: DATA WRANGLING

# Load the raw FBref player dataset. The raw CSV is not edited directly.
# Inspect the raw dataset before cleaning.

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


# Keep only the variables needed for the fouls analysis.
# Rename them to simpler names for the rest of the script.
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


# Convert playing time and fouls to numeric values.
# Unreadable values become missing rather than being guessed.
df['nineties_played'] = pd.to_numeric(df['nineties_played'], errors='coerce')
df['fouls_committed'] = pd.to_numeric(df['fouls_committed'], errors='coerce')

print('\nMissing values in selected columns (after numeric coercion):')
print(df[['player', 'position', 'team', 'nineties_played', 'fouls_committed']].isna().sum())


# Clean the FBref team names by removing leading country codes.
def clean_team(name):
    if pd.isna(name):
        return name
    s = str(name).strip()
    s = re.sub(r'^[A-Za-z]{2,3}\s+', '', s)
    return s.strip()

# Work on a cleaned copy so the original selection remains available.
df_clean = df.copy()
df_clean['team'] = df_clean['team'].apply(clean_team)

print('\nSample cleaned team values:')
print(df_clean['team'].dropna().unique()[:20])


# Check missing values in the fields needed for the analysis.
# Report missing fields before filtering the eligible population.
required = ['position', 'nineties_played', 'fouls_committed']
missing_required_mask = df_clean[required].isna().any(axis=1)
missing_required_count = missing_required_mask.sum()
print(f"\nRows with missing required fields (position, nineties_played, fouls_committed): {missing_required_count}")
if missing_required_count > 0:
    print('Examples of rows with missing required fields:')
    print(df_clean[missing_required_mask].head(5))


# Define the eligible population.
# Keep outfield players with at least 90 minutes played.
# The 90-minute rule avoids comparisons based on very little playing time.
players_with_1plus_90s = df_clean[df_clean['nineties_played'] >= 1.0].shape[0]

df_eligible = df_clean.dropna(subset=required).copy()

df_eligible = df_eligible[df_eligible['position'] != 'GK'].copy()

df_eligible = df_eligible[df_eligible['nineties_played'] >= 1.0].copy()

print('\n-- POPULATION COUNTS & FREQUENCIES --')
print('Raw dataset rows:', len(df_raw))
print('Players with >= 1.0 90s (before excluding GKs):', players_with_1plus_90s)
print('Final eligible outfield population size:', len(df_eligible))
print('\nFrequency table of positions (eligible population):')
print(df_eligible['position'].value_counts(dropna=False))


# Create the variables used in the analysis.
# committed_foul shows whether a player committed at least one foul.
# fouls_per_90 adjusts the number of fouls for playing time.
df_eligible['committed_foul'] = np.where(df_eligible['fouls_committed'] > 0, 1,
                                         np.where(df_eligible['fouls_committed'] == 0, 0, np.nan))

df_eligible['fouls_per_90'] = df_eligible['fouls_committed'] / df_eligible['nineties_played']


# Validate the eligible population and constructed variables before analysis.
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

print('\nData wrangling complete: eligible population and features are ready.')

# SKILL 3: SAMPLING
# Take a simple random sample of 250 eligible players.
# random_state=42 makes the sample reproducible.
sample = df_eligible.sample(n=250, random_state=42)

print('\nSKILL 3: SAMPLING')
print('Sample size:', len(sample))
print('\nSample position frequency table:')
print(sample['position'].value_counts(dropna=False))
print('\nNumber of exact DF players in sample:', int((sample['position'] == 'DF').sum()))
print('Number of exact FW players in sample:', int((sample['position'] == 'FW').sum()))


# SKILL 4: DESCRIPTIVE STATISTICS
# Calculate the sample proportion who committed at least one foul.
print('\nSKILL 4: DESCRIPTIVE STATISTICS')
num_committed_sample = int(sample['committed_foul'].sum())
num_zero_sample = int((sample['committed_foul'] == 0).sum())
p_hat = num_committed_sample / len(sample)
print('\nProportion analysis (sample of 250):')
print('Number who committed at least one foul:', num_committed_sample)
print('Number who committed zero fouls:', num_zero_sample)
print('Sample proportion (p_hat):', p_hat)
print('Sample percentage who committed at least one foul:', f"{p_hat*100:.2f}%")


# Use exact defenders and forwards for the position comparison.
defenders = sample[sample['position'] == 'DF']['fouls_per_90'].dropna()
forwards = sample[sample['position'] == 'FW']['fouls_per_90'].dropna()

def print_stats(name, series):
    n = len(series)
    mean = series.mean() if n > 0 else float('nan')
    median = series.median() if n > 0 else float('nan')
    std = series.std(ddof=1) if n > 1 else float('nan')
    mn = series.min() if n > 0 else float('nan')
    mx = series.max() if n > 0 else float('nan')
    print(f"\n{name} (n={n})")
    print(' mean fouls_per_90:', mean)
    print(' median fouls_per_90:', median)
    print(' std (sample):', std)
    print(' min:', mn)
    print(' max:', mx)

# Calculate descriptive statistics for both groups.
print('\nDefender vs Forward descriptive statistics (sample)')
print_stats('Defenders (DF)', defenders)
print_stats('Forwards (FW)', forwards)

mean_def = defenders.mean() if len(defenders) > 0 else float('nan')
mean_fwd = forwards.mean() if len(forwards) > 0 else float('nan')
print('\nDifference in means (defender mean - forward mean):', mean_def - mean_fwd)


# SKILL 5 & 6: CONFIDENCE INTERVAL & TWO-SAMPLE T-TEST
# Calculate the 95% confidence interval for the population proportion.

print('\nSKILL 5 & 6: CONFIDENCE INTERVAL & TWO-SAMPLE T-TEST')
n = len(sample)
print('\nProportion 95% CI (sample-based)')
print(' sample size n:', n)
print(' p_hat:', p_hat)
# Check the success/failure condition for the proportion confidence interval.
print(' n * p_hat:', n * p_hat)
print(' n * (1 - p_hat):', n * (1 - p_hat))

standard_error = np.sqrt(p_hat * (1 - p_hat) / n)
z_critical = 1.96
lower_ci = p_hat - z_critical * standard_error
upper_ci = p_hat + z_critical * standard_error
print(' standard error:', standard_error)
print(' 95% CI (proportions):', (lower_ci, upper_ci))
print(' 95% CI (percent):', (f"{lower_ci*100:.2f}%", f"{upper_ci*100:.2f}%"))


# Hypotheses:
# H0: mu_DF = mu_FW
# Ha: mu_DF != mu_FW
# Use a two-sided Welch t-test with alpha = 0.05.
alpha = 0.05
print('\nWelch two-sample t-test (defenders vs forwards)')
print('Using exact DF and exact FW from the sample only')
mean_def_sample = mean_def
mean_fwd_sample = mean_fwd
print(' defender sample mean:', mean_def_sample)
print(' forward sample mean:', mean_fwd_sample)
print(' difference in sample means (def - fwd):', mean_def_sample - mean_fwd_sample)

t_res = stats.ttest_ind(defenders, forwards, equal_var=False)
t_stat = float(t_res.statistic)
p_value = float(t_res.pvalue)
print(' t-statistic:', t_stat)
print(' p-value:', p_value)
print(' alpha:', alpha)
if p_value < alpha:
    print(' Decision: reject H0 because p-value < 0.05')
    print(' Conclusion: There is sufficient evidence at the 5% significance level to conclude that defenders and forwards differ in their population mean fouls per 90.')
else:
    print(' Decision: do not reject H0 because p-value > 0.05')
    print(' Conclusion: There is insufficient evidence at the 5% significance level to conclude that defenders and forwards differ in their population mean fouls per 90.')


# VISUALISATIONS
# Plot the percentage of sampled players who committed at least one foul.
counts_at_least_one = int((sample['committed_foul'] == 1).sum())
counts_zero = int((sample['committed_foul'] == 0).sum())
labels = ["At least one foul", "Zero fouls"]
percent_at_least_one = counts_at_least_one / len(sample) * 100
percent_zero = counts_zero / len(sample) * 100
percents = [percent_at_least_one, percent_zero]

plt.figure(figsize=(6,4))
bars = plt.bar(labels, percents)
plt.title('Foul Commitment Among Sampled Outfield Players')
plt.ylabel('Percentage of Players (%)')
# Label each bar with its percentage.
for bar, pct in zip(bars, percents):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1.0, f"{pct:.1f}%", ha='center', va='bottom')
plt.tight_layout()
plt.savefig('task_fouls/foul_proportion.png')
plt.close()


# Compare the distribution of fouls per 90 for defenders and forwards.
plt.figure(figsize=(6,5))
data = [defenders, forwards]
plt.boxplot(data)
plt.xticks([1, 2], ['Defenders', 'Forwards'])
plt.title('Distribution of Fouls per 90: Defenders vs Forwards')
plt.ylabel('Fouls per 90 Minutes')
# The diamond markers represent the sample means.
mean_def_val = defenders.mean()
mean_fwd_val = forwards.mean()
plt.scatter([1], [mean_def_val], color='red', marker='D', s=60, zorder=5)
plt.scatter([2], [mean_fwd_val], color='red', marker='D', s=60, zorder=5)
# Add a legend entry for the sample mean marker.
from matplotlib.lines import Line2D
proxy = Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=8)
plt.legend([proxy], ['Sample mean'], loc='upper right')
plt.tight_layout()
plt.savefig('task_fouls/defender_forward_fouls.png')
plt.close()


