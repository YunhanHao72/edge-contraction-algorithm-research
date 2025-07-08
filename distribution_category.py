# final step to calculate average improvement for each category
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv_dir = "result_csvs_uni_knott"
output_dir = "result_category_uni_knott"
os.makedirs(output_dir, exist_ok=True)

round_to_integer = True 
decimal_precision = 6

for subfolder in sorted(os.listdir(csv_dir)):
    subfolder_path = os.path.join(csv_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    print(f"Processing category: {subfolder}")

    all_greedy_norm = []
    all_lookahead_norm = []

    for fname in sorted(os.listdir(subfolder_path)):
        if not fname.endswith("_greedy_joining.csv"):
            continue

        instance_name = fname.replace("_greedy_joining.csv", "")
        greedy_csv = os.path.join(subfolder_path, fname)
        lookahead_csv = os.path.join(subfolder_path, f"{instance_name}_greedy_joining_lookahead.csv")

        if not os.path.exists(lookahead_csv):
            print(f"Missing for {instance_name}, skip.")
            continue

        greedy_df = pd.read_csv(greedy_csv)
        lookahead_df = pd.read_csv(lookahead_csv)

        if round_to_integer:
            greedy_df["CleanCost"] = np.round(greedy_df["Cost"]).astype(int)
            lookahead_df["CleanCost"] = np.round(lookahead_df["Cost"]).astype(int)
        else:
            greedy_df["CleanCost"] = np.round(greedy_df["Cost"], decimals=decimal_precision)
            lookahead_df["CleanCost"] = np.round(lookahead_df["Cost"], decimals=decimal_precision)

        greedy_mean = greedy_df["CleanCost"].mean()
        if greedy_mean == 0:
            print(f"Zero mean for {instance_name}, skip normalization.")
            continue

        greedy_norm = 100 * (greedy_df["CleanCost"] - greedy_mean) / abs(greedy_mean)
        lookahead_norm = 100 * (lookahead_df["CleanCost"] - greedy_mean) / abs(greedy_mean)

        all_greedy_norm.extend(greedy_norm)
        all_lookahead_norm.extend(lookahead_norm)

    if len(all_greedy_norm) == 0 or len(all_lookahead_norm) == 0:
        print(f"No valid data in {subfolder}, skip category plot.\n")
        continue

    plt.figure(figsize=(6, 4))
    sns.kdeplot(all_greedy_norm, fill=True, label="GAEC")
    sns.kdeplot(all_lookahead_norm, fill=True, label="Lookahead Greedy Joining")

    plt.axvline(np.mean(all_greedy_norm), color="blue", linestyle="--", label=f"greedy mean: {np.mean(all_greedy_norm):.2f}%")
    plt.axvline(np.mean(all_lookahead_norm), color="orange", linestyle="--", label=f"lookahead mean: {np.mean(all_lookahead_norm):.2f}%")

    plt.title(f"Normalized Cost Improvement: {subfolder}")
    plt.xlabel("Improvement(%)")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{subfolder}_kde.png")
    plt.savefig(save_path)
    plt.close()

    avg_improvement = np.mean(all_lookahead_norm) - np.mean(all_greedy_norm)
    print(f"{subfolder} avg improvement over greedy: {avg_improvement:.2f}%\n")