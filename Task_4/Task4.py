"""
SKILL 1: QUESTION FORMULATION
Analytic Question: Do teams with a faster average top speed score significantly 
more goals per match than teams with a slower average top speed in the 2026 World Cup?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# SKILL 2: DATA WRANGLING (Acquisition, Cleaning, & Feature Construction)

# 1. Load the raw datasets using Pandas
df_speed = pd.read_csv('Raw_Dataset_FIFA_Player_Stats(Sheet1).csv')
df_goals = pd.read_csv('Raw_Dataset_FIFA_Player_Stats(Sheet2).csv')

# 2. Reshape the messy 5-row blocks into clean DataFrames using .iloc stepping
# This automates the extraction of player names and stats without manual Excel work.

clean_speed = pd.DataFrame({
    'Team': df_speed.iloc[3::5].reset_index(drop=True)['Player'],
    'Top_Speed': pd.to_numeric(df_speed.iloc[0::5].reset_index(drop=True)['Unnamed: 2'], errors='coerce')
})

clean_goals = pd.DataFrame({
    'Team': df_goals.iloc[3::5].reset_index(drop=True)['Player'],
    'Goals': pd.to_numeric(df_goals.iloc[0::5].reset_index(drop=True)['Unnamed: 2'], errors='coerce')
})

# 3. Filter inactive players (Case Deletion)
# Retaining 0.0 top speeds would artificially deflate the team's true physical average.
active_players = clean_speed[clean_speed['Top_Speed'] > 0].copy()

# 4. Feature Construction (Grouping and Aggregating)
# We must convert individual player stats into team-level metrics to test our hypothesis.
# .groupby() groups nominal categories together, .agg() applies the mathematical function.
team_stats = active_players.groupby('Team').agg(Avg_Team_Top_Speed=('Top_Speed', 'mean')).reset_index()
team_goals = clean_goals.groupby('Team').agg(Total_Goals=('Goals', 'sum')).reset_index()

# 5. Inner Join
# .merge() staples our two cleaned datasets together by matching the 'Team' column.
master_df = pd.merge(team_stats, team_goals, on='Team', how='inner')

# SKILL 3: SAMPLING (Data Preparation & Median Split)

# 1. Calculate the median top speed
# The median provides an objective mathematical threshold to split our 48 teams in half.
median_speed = master_df['Avg_Team_Top_Speed'].median()

# 2. Categorize the teams using np.where
# np.where(condition, true_value, false_value) creates our two distinct comparison groups.
master_df['Speed_Category'] = np.where(master_df['Avg_Team_Top_Speed'] > median_speed, 'Faster', 'Slower')

# Isolate the goals for each group into distinct Series for our t-test
faster_goals = master_df[master_df['Speed_Category'] == 'Faster']['Total_Goals']
slower_goals = master_df[master_df['Speed_Category'] == 'Slower']['Total_Goals']

# SKILL 4: DESCRIPTIVE STATISTICS

# Generate a histogram using Matplotlib
# Because our sample size is under 30 (n=24 per group), we must visually check for normality.
plt.hist(faster_goals, alpha=0.5, label=f'Faster Teams (n={len(faster_goals)})', bins=10)
plt.hist(slower_goals, alpha=0.5, label=f'Slower Teams (n={len(slower_goals)})', bins=10)
plt.title('Distribution of Team Goals by Speed Category')
plt.xlabel('Total Goals')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Print central tendencies
print(f"Faster Teams Mean Goals: {faster_goals.mean():.2f}")
print(f"Slower Teams Mean Goals: {slower_goals.mean():.2f}")

# SKILL 5 & 6: CONFIDENCE INTERVAL & TWO-SAMPLE T-TEST

# 1. Two-Sample t-Test using scipy.stats
# Why: ttest_ind() compares the means of two independent groups. equal_var=False runs Welch's t-test.
t_stat, p_value = stats.ttest_ind(faster_goals, slower_goals, equal_var=False)

# 2. Calculate the 95% Confidence Interval for the difference in means
# This estimates the true population parameter for how many extra goals a faster team scores.
mean_diff = faster_goals.mean() - slower_goals.mean()
se_diff = np.sqrt(faster_goals.var(ddof=1)/len(faster_goals) + slower_goals.var(ddof=1)/len(slower_goals))
df = min(len(faster_goals)-1, len(slower_goals)-1) # Degrees of freedom approximation
t_crit = stats.t.ppf(0.975, df) # Critical t-value for 95% CI
ci_lower = mean_diff - (t_crit * se_diff)
ci_upper = mean_diff + (t_crit * se_diff)

print("\nINFERENTIAL STATISTICS")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"95% Confidence Interval for Mean Difference: [{ci_lower:.2f}, {ci_upper:.2f}]")

