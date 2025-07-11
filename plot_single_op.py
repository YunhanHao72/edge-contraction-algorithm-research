# drawing the plots for each instance
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, MaxNLocator, FormatStrFormatter

csv_root = "result_csvs_lookahead"
output_dir = "plots_lookahead"
os.makedirs(output_dir, exist_ok=True)

round_to_integer = True 
decimal_precision = 6 

total_tasks = 0
completed_tasks = 0

for subdir in os.listdir(csv_root):
    subdir_path = os.path.join(csv_root, subdir)
    if not os.path.isdir(subdir_path):
        continue
    for fname in os.listdir(subdir_path):
        if "_greedy_joining.csv" in fname:
            total_tasks += 1

print(f"Total instances to process: {total_tasks}\n")

for subdir in sorted(os.listdir(csv_root)):
    subdir_path = os.path.join(csv_root, subdir)
    if not os.path.isdir(subdir_path):
        continue

    for fname in sorted(os.listdir(subdir_path)):
        if not fname.endswith(".csv") or "_greedy_joining.csv" not in fname:
            continue

        instance_name = fname.replace("_greedy_joining.csv", "")
        greedy_csv = os.path.join(subdir_path, fname)
        lookahead_csv = os.path.join(subdir_path, f"{instance_name}_greedy_joining_lookahead.csv")

        if not os.path.exists(lookahead_csv):
            print(f"Missing file for {instance_name}, skipping.")
            continue

        print(f"Processing {subdir}/{instance_name} ...")

        greedy_df = pd.read_csv(greedy_csv)
        lookahead_df = pd.read_csv(lookahead_csv)

        if round_to_integer:
            greedy_df["RoundedCost"] = np.round(greedy_df["Cost"]).astype(int)
            lookahead_df["RoundedCost"] = np.round(lookahead_df["Cost"]).astype(int)
        else:
            greedy_df["RoundedCost"] = np.round(greedy_df["Cost"], decimals=decimal_precision)
            lookahead_df["RoundedCost"] = np.round(lookahead_df["Cost"], decimals=decimal_precision)

        greedy_median = greedy_df["Cost"].median()
        lookahead_median = lookahead_df["Cost"].median()

        print(f"Median Cost → greedy_joining: {greedy_median:.2f}, "
              f"lookahead: {lookahead_median:.2f}")

        plt.figure(figsize=(6, 4))
        sns.kdeplot(greedy_df["RoundedCost"], label="GAEC", fill=True)
        sns.kdeplot(lookahead_df["RoundedCost"], label="Lookahead Greedy Joining", fill=True)

        plt.axvline(greedy_median, color='blue', linestyle='--', linewidth=1)
        plt.axvline(lookahead_median, color='orange', linestyle='--', linewidth=1)

        plt.text(greedy_median, plt.ylim()[1]*0.9,
                 f"{greedy_median:.2f}", color='blue', ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))

        plt.text(lookahead_median, plt.ylim()[1]*0.8,
                 f"{lookahead_median:.2f}", color='orange', ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='orange', boxstyle='round,pad=0.3'))

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')

        plt.title(f"Cost - {instance_name}")
        plt.xlabel("Cost")
        plt.ylabel("Density")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        save_name = f"{subdir}_{instance_name}_kde.png"
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path)
        plt.close()

        completed_tasks += 1
        print(f"Done ({completed_tasks}/{total_tasks}) → Saved to: {save_path}\n")