# Task: Interceptions Analysis
# FIFA World Cup 2026

# Analytic question:
# Among outfield players who played at least 90 minutes at the
# FIFA World Cup 2026, what was the mean number of interceptions
# per 90 minutes, and did defenders and midfielders differ
# significantly in their mean interceptions per 90 minutes?


import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Load the raw FBref dataset
df = pd.read_csv(
    "Task Interceptions/fifa2026_player_misc_raw.csv",
    header=1
)

# Display the first five rows
print("FIRST FIVE ROWS")
print(df.head())


# Check the size of the dataset
print("\nDATASET SIZE")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


# Display column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())


# Check the data types
print("\nDATA TYPES")
print(df.dtypes)


# ============================================================
# TASK 2: DATA WRANGLING
# ============================================================

# Keep only the columns needed for this analytic question
data = df[
    ["Player", "Pos", "Squad", "90s", "Int"]
].copy()


print("\nRELEVANT DATA")
print(data.head())


# Check the size after selecting relevant columns
print("\nSIZE AFTER SELECTING RELEVANT COLUMNS")
print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])


# Check for missing values in the relevant columns
print("\nMISSING VALUES")
print(data.isnull().sum())


# Check for duplicate player records
duplicate_count = data.duplicated(
    subset=["Player", "Squad"]
).sum()

print("\nDUPLICATE PLAYER RECORDS")
print("Number of duplicates:", duplicate_count)


# Display all position categories
print("\nPOSITION CATEGORIES")
print(data["Pos"].value_counts())


# Check the range of the two numerical variables
print("\n90s RANGE")
print("Minimum:", data["90s"].min())
print("Maximum:", data["90s"].max())

print("\nINTERCEPTIONS RANGE")
print("Minimum:", data["Int"].min())
print("Maximum:", data["Int"].max())


# ============================================================
# TASK 3: DATA PREPARATION
# ============================================================

# Remove goalkeepers because the analytic question
# only considers outfield players
outfield = data[
    data["Pos"] != "GK"
].copy()


print("\nOUTFIELD PLAYERS")
print("Number of outfield players:", len(outfield))


# Keep only players who played at least 90 minutes
# In FBref, 90s = 1.0 means 90 minutes played
eligible = outfield[
    outfield["90s"] >= 1.0
].copy()


print("\nELIGIBLE PLAYERS AFTER 90-MINUTE FILTER")
print("Number of eligible players:", len(eligible))


# Check the position categories after filtering
print("\nPOSITIONS AFTER FILTERING")
print(eligible["Pos"].value_counts())


# Create the main analysis variable:
# interceptions per 90 minutes
eligible["Interceptions_per_90"] = (
    eligible["Int"] / eligible["90s"]
)


# Display the first few prepared rows
print("\nPREPARED DATA")
print(
    eligible[
        [
            "Player",
            "Pos",
            "Squad",
            "90s",
            "Int",
            "Interceptions_per_90"
        ]
    ].head(10)
)


# Check that the new variable has no missing or infinite values
print("\nCHECK INTERCEPTIONS PER 90")

print(
    "Missing values:",
    eligible["Interceptions_per_90"].isnull().sum()
)

print(
    "Minimum:",
    eligible["Interceptions_per_90"].min()
)

print(
    "Maximum:",
    eligible["Interceptions_per_90"].max()
)


# ============================================================
# TASK 4: SIMPLE RANDOM SAMPLING
# ============================================================

# Define the eligible population
population = eligible.copy()

print("\nPOPULATION SIZE")
print("Number of eligible players:", len(population))


# Set the sample size
sample_size = 300


# Take a simple random sample without replacement
sample = population.sample(
    n=sample_size,
    random_state=42,
    replace=False
)


print("\nSAMPLE SIZE")
print("Number of sampled players:", len(sample))


# Display the first 10 sampled players
print("\nFIRST 10 SAMPLED PLAYERS")
print(
    sample[
        [
            "Player",
            "Pos",
            "Squad",
            "90s",
            "Int",
            "Interceptions_per_90"
        ]
    ].head(10)
)


# Check how positions are represented in the sample
print("\nPOSITION COUNTS IN SAMPLE")
print(sample["Pos"].value_counts())


# Save the sample as a new CSV file
sample.to_csv(
    "Task Interceptions/interceptions_sample.csv",
    index=False
)

print("\nSample saved as interceptions_sample.csv")



# ============================================================
# TASK 5: DESCRIPTIVE STATISTICS
# ============================================================

# Select the main variable for analysis
interceptions_per_90 = sample["Interceptions_per_90"]


# Calculate descriptive statistics
sample_mean = interceptions_per_90.mean()
sample_median = interceptions_per_90.median()
sample_std = interceptions_per_90.std()
sample_min = interceptions_per_90.min()
sample_max = interceptions_per_90.max()

q1 = interceptions_per_90.quantile(0.25)
q3 = interceptions_per_90.quantile(0.75)


# Display the results
print("\nDESCRIPTIVE STATISTICS")
print("----------------------")

print("Sample size:", len(interceptions_per_90))
print("Mean:", round(sample_mean, 3))
print("Median:", round(sample_median, 3))
print("Standard deviation:", round(sample_std, 3))
print("Minimum:", round(sample_min, 3))
print("First quartile (Q1):", round(q1, 3))
print("Third quartile (Q3):", round(q3, 3))
print("Maximum:", round(sample_max, 3))


# ============================================================
# TASK 6: HISTOGRAM
# ============================================================

# Create a histogram to examine the distribution
# of interceptions per 90 minutes

plt.figure(figsize=(8, 5))

plt.hist(
    sample["Interceptions_per_90"],
    bins=20
)

plt.xlabel("Interceptions per 90 minutes")
plt.ylabel("Number of players")
plt.title("Distribution of Interceptions per 90 Minutes")

plt.tight_layout()


# Save the graph in the task folder
plt.savefig(
    "Task Interceptions/interceptions_histogram.png",
    dpi=300
)


# Display the graph
plt.show()


# ============================================================
# TASK 7: 95% CONFIDENCE INTERVAL FOR THE MEAN
# ============================================================

# Number of observations in the sample
n = len(interceptions_per_90)


# Sample mean
sample_mean = interceptions_per_90.mean()


# Sample standard deviation
sample_std = interceptions_per_90.std(ddof=1)


# Calculate the standard error
standard_error = sample_std / np.sqrt(n)


# Calculate the 95% confidence interval
confidence_interval = stats.t.interval(
    confidence=0.95,
    df=n - 1,
    loc=sample_mean,
    scale=standard_error
)


print("\n95% CONFIDENCE INTERVAL")
print("-----------------------")

print("Sample size:", n)
print("Sample mean:", round(sample_mean, 3))
print("Standard error:", round(standard_error, 3))

print(
    "95% confidence interval:",
    round(confidence_interval[0], 3),
    "to",
    round(confidence_interval[1], 3)
)


# ============================================================
# TASK 8: PREPARE DEFENDER AND MIDFIELDER GROUPS
# ============================================================

# Select defenders from the random sample
defenders = sample[
    sample["Pos"] == "DF"
]["Interceptions_per_90"]


# Select midfielders from the random sample
midfielders = sample[
    sample["Pos"] == "MF"
]["Interceptions_per_90"]


# Display the number of players in each group
print("\nDEFENDER AND MIDFIELDER GROUPS")
print("-------------------------------")

print("Number of defenders:", len(defenders))
print("Number of midfielders:", len(midfielders))


# Calculate descriptive statistics for defenders
defender_mean = defenders.mean()
defender_median = defenders.median()
defender_std = defenders.std(ddof=1)


# Calculate descriptive statistics for midfielders
midfielder_mean = midfielders.mean()
midfielder_median = midfielders.median()
midfielder_std = midfielders.std(ddof=1)


# Display the group statistics
print("\nDEFENDER STATISTICS")
print("Mean:", round(defender_mean, 3))
print("Median:", round(defender_median, 3))
print("Standard deviation:", round(defender_std, 3))


print("\nMIDFIELDER STATISTICS")
print("Mean:", round(midfielder_mean, 3))
print("Median:", round(midfielder_median, 3))
print("Standard deviation:", round(midfielder_std, 3))


# Calculate the observed difference between the two sample means
mean_difference = defender_mean - midfielder_mean

print("\nOBSERVED MEAN DIFFERENCE")
print(
    "Defender mean - Midfielder mean:",
    round(mean_difference, 3)
)

# ============================================================
# TASK 9: DEFENDER VS MIDFIELDER BOXPLOT
# ============================================================

# Keep only defenders and midfielders for the comparison
comparison_data = sample[
    sample["Pos"].isin(["DF", "MF"])
].copy()


# Create the boxplot
plt.figure(figsize=(7, 5))

comparison_data.boxplot(
    column="Interceptions_per_90",
    by="Pos"
)

plt.xlabel("Position")
plt.ylabel("Interceptions per 90 minutes")
plt.title("Interceptions per 90: Defenders vs Midfielders")

# Remove the automatic pandas subtitle
plt.suptitle("")

plt.tight_layout()


# Save the graph
plt.savefig(
    "Task Interceptions/defender_midfielder_interceptions.png",
    dpi=300
)


# Display the graph
plt.show()


# ============================================================
# TASK 10: TWO-SAMPLE T-TEST
# ============================================================

# Hypotheses:
#
# H0: The mean interceptions per 90 minutes is the same
#     for defenders and midfielders.
#
# H1: The mean interceptions per 90 minutes is different
#     for defenders and midfielders.


# Set significance level
alpha = 0.05


# Perform Welch's independent two-sample t-test
t_statistic, p_value = stats.ttest_ind(
    defenders,
    midfielders,
    equal_var=False
)


print("\nTWO-SAMPLE T-TEST")
print("-----------------")

print("Defender mean:", round(defender_mean, 3))
print("Midfielder mean:", round(midfielder_mean, 3))

print(
    "Mean difference:",
    round(defender_mean - midfielder_mean, 3)
)

print(
    "t-statistic:",
    round(t_statistic, 3)
)

print(
    "p-value:",
    round(p_value, 4)
)


# Make the hypothesis-test decision
if p_value < alpha:

    print("\nDecision: Reject the null hypothesis.")

    print(
        "There is statistically significant evidence "
        "that defenders and midfielders differ in their "
        "mean interceptions per 90 minutes."
    )

else:

    print("\nDecision: Fail to reject the null hypothesis.")

    print(
        "There is not enough statistical evidence "
        "to conclude that defenders and midfielders differ "
        "in their mean interceptions per 90 minutes."
    )

    # ============================================================
# TASK 11: FINAL SUMMARY
# ============================================================

print("\nFINAL ANALYSIS SUMMARY")
print("----------------------")

print(
    f"The sample mean was "
    f"{sample_mean:.3f} interceptions per 90 minutes."
)

print(
    f"The 95% confidence interval for the mean was "
    f"{confidence_interval[0]:.3f} to "
    f"{confidence_interval[1]:.3f}."
)

print(
    f"Defenders recorded a mean of "
    f"{defender_mean:.3f} interceptions per 90, "
    f"while midfielders recorded "
    f"{midfielder_mean:.3f}."
)

print(
    f"The Welch two-sample t-test produced "
    f"a p-value of {p_value:.4f}."
)

if p_value < 0.05:
    print(
        "The difference between defenders and midfielders "
        "was statistically significant."
    )
else:
    print(
        "The difference between defenders and midfielders "
        "was not statistically significant."
    )