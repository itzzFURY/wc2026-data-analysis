import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.stats.weightstats as stm
import math

# ==============================================================================
# STEP 1: QUESTION FORMULATION
# Do teams with a faster average top speed score significantly 
# more goals per match than teams with a slower average top speed in the 2026 World Cup?
# ==============================================================================

# ==============================================================================
# STEP 2: DATA WRANGLING 
# ==============================================================================

# Load the raw datasets
df_speed = pd.read_csv('Raw_Dataset_FIFA_Player_Stats(Sheet1).csv')
df_goals = pd.read_csv('Raw_Dataset_FIFA_Player_Stats(Sheet2).csv')

# Semi-automated extraction using .iloc stepping
clean_speed = pd.DataFrame({
    'Team': df_speed.iloc[3::5].reset_index(drop=True)['Player'],
    'Top_Speed': pd.to_numeric(df_speed.iloc[0::5].reset_index(drop=True)['Unnamed: 2'], errors='coerce')
})

clean_goals = pd.DataFrame({
    'Team': df_goals.iloc[3::5].reset_index(drop=True)['Player'],
    'Goals': pd.to_numeric(df_goals.iloc[0::5].reset_index(drop=True)['Unnamed: 2'], errors='coerce')
})

# Filter inactive players (Case Deletion)
active_players = clean_speed[clean_speed['Top_Speed'] > 0].copy()

# Feature Construction
team_stats = active_players.groupby('Team').agg(Avg_Team_Top_Speed=('Top_Speed', 'mean')).reset_index()
team_goals = clean_goals.groupby('Team').agg(Total_Goals=('Goals', 'sum')).reset_index()

# Combining Datasets 
master_df = team_stats.join(
    team_goals.set_index(["Team"]),
    on=["Team"]
)

# ==============================================================================
# STEP 3: DATA PRREPARATION AND SAMPLING
# ==============================================================================

median_speed = master_df['Avg_Team_Top_Speed'].median()
master_df['Speed_Category'] = np.where(master_df['Avg_Team_Top_Speed'] > median_speed, 'Faster', 'Slower')

faster_goals = master_df[master_df['Speed_Category'] == 'Faster']['Total_Goals'].to_numpy()
slower_goals = master_df[master_df['Speed_Category'] == 'Slower']['Total_Goals'].to_numpy()

# ==============================================================================
# STEP 4: DESCRIPTIVE STATISTICS
# ==============================================================================

x_bar_fast = st.tmean(faster_goals)
s_fast = st.tstd(faster_goals)
n_fast = len(faster_goals)

x_bar_slow = st.tmean(slower_goals)
s_slow = st.tstd(slower_goals)
n_slow = len(slower_goals)

print("Faster Teams - Mean: %.2f, Std Dev: %.2f, Size: %d" % (x_bar_fast, s_fast, n_fast))
print("Slower Teams - Mean: %.2f, Std Dev: %.2f, Size: %d" % (x_bar_slow, s_slow, n_slow))

min_val = min(master_df['Total_Goals'])
max_val = max(master_df['Total_Goals'])
the_range = max_val - min_val
bin_width = 2 # As the goals range is small, a width of 2 is appropriate
bin_count = int(the_range/bin_width) if the_range > 0 else 5

plt.hist(faster_goals, alpha=0.5, label='Faster Teams', bins=bin_count)
plt.hist(slower_goals, alpha=0.5, label='Slower Teams', bins=bin_count)
plt.title('Distribution of Team Goals by Speed Category')
plt.xlabel('Total Goals')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# ==============================================================================
# STEP 5 & 6: INFERENTIAL STATISTICS (Confidence Interval & Two Sample t-Test)
# ==============================================================================

t_stats, p_val = st.ttest_ind_from_stats(
    x_bar_fast, s_fast, n_fast, 
    x_bar_slow, s_slow, n_slow, 
    equal_var=False, 
    alternative='two-sided'
)

mean_diff = x_bar_fast - x_bar_slow
std_err_diff = math.sqrt((s_fast**2 / n_fast) + (s_slow**2 / n_slow))
df = min(n_fast - 1, n_slow - 1)

ci_low_stm, ci_upp_stm = stm._tconfint_generic(
    mean_diff, std_err_diff, df, alpha=0.05, alternative="two-sided"
)

print("\nINFERENTIAL STATISTICS")
print("t-statistic (t*): %.3f" % t_stats)
print("p-value: %.4f" % p_val)
print("95%% C.I. of the mean difference is between %.2f and %.2f." % (ci_low_stm, ci_upp_stm))