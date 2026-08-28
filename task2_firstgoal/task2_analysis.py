import pandas as pd
import scipy.stats as st
import statsmodels.stats.proportion as stm
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv("task2_final_dataset.csv")
print("Total matches:", len(df))
print("Overall win rate: %.3f" % df["won"].mean())

early = df[df["timing_group"] == "early"]
late = df[df["timing_group"] == "late"]
print("Early: n=%d rate=%.3f" % (len(early), early["won"].mean()))
print("Late:  n=%d rate=%.3f" % (len(late), late["won"].mean()))

# this helps us decide where to split the two groups
sns.histplot(df["first_goal_min"], bins=18)
plt.axvline(df["first_goal_min"].median(), color="red", label="median")
plt.title("First goal minute distribution")
plt.xlabel("minute")
plt.legend()
plt.savefig("first_goal_hist.png")
plt.close()
print("median first goal minute:", df["first_goal_min"].median())

# tried splitting by half first but 2nd half was too small (under 30)
first_half = df[df["half"] == "1st"]
second_half = df[df["half"] == "2nd"]
print("Halftime split -> 1st:%d 2nd:%d" % (len(first_half), len(second_half)))

SEED = 42
ci_sample = df.sample(n=40, random_state=SEED)
early_sample = early.sample(n=40, random_state=SEED)
late_sample = late.sample(n=40, random_state=SEED)

wins = ci_sample["won"].sum()
total = len(ci_sample)
p_hat = wins / total
print("Check: np=%.1f n(1-p)=%.1f" % (total*p_hat, total*(1-p_hat)))

ci_low, ci_upp = stm.proportion_confint(wins, total, alpha=0.05, method="normal")
print("95%% CI: %.3f to %.3f" % (ci_low, ci_upp))

# two-sample t-test, early vs late
x_bar1, s1, n1 = st.tmean(early_sample["won"]), st.tstd(early_sample["won"]), len(early_sample)
x_bar2, s2, n2 = st.tmean(late_sample["won"]), st.tstd(late_sample["won"]), len(late_sample)

t_stats, p_val = st.ttest_ind_from_stats(x_bar1, s1, n1, x_bar2, s2, n2,
                                         equal_var=False, alternative="two-sided")
print("t = %.3f  p = %.4f" % (t_stats, p_val))

if p_val < 0.05:
    print("Reject H0 - win rates differ")
else:
    print("Do not reject H0 - no significant difference")

groups = ["early", "late"]
rates = [early["won"].mean(), late["won"].mean()]
plt.bar(groups, rates, color=["skyblue", "orange"])
plt.title("Win rate: early vs late first goal")
plt.ylabel("win rate")
plt.ylim(0, 1)
plt.savefig("win_rate_bar.png")
plt.close()