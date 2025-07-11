import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv_dir = "result_csvs_lookahead/random"
output_dir = "result_random_subclass_lookahead"
os.makedirs(output_dir, exist_ok=True)

round_to_integer = True
decimal_precision = 6

summary_results = []

def detect_subclass(instance_name):
    name = instance_name.lower()
    if "cpn" in name:
        return "CPn"
    elif "unif" in name:
        return "unif"
    elif "b" in name:
        return "b"
    elif "p" in name:
        return "p"
    else:
        return "rand"

subclass_data = {}

for fname in sorted(os.listdir(csv_dir)):
    if not fname.endswith("_greedy_joining.csv"):
        continue

    instance_name = fname.replace("_greedy_joining.csv", "")
    subclass = detect_subclass(instance_name)

    greedy_csv = os.path.join(csv_dir, fname)
    lookahead_csv = os.path.join(csv_dir, f"{instance_name}_greedy_joining_lookahead.csv")

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

    if subclass not in subclass_data:
        subclass_data[subclass] = {
            "greedy": [],
            "lookahead": []
        }

    subclass_data[subclass]["greedy"].extend(greedy_norm)
    subclass_data[subclass]["lookahead"].extend(lookahead_norm)

for subclass, data in subclass_data.items():
    greedy_vals = data["greedy"]
    lookahead_vals = data["lookahead"]

    if len(greedy_vals) == 0 or len(lookahead_vals) == 0:
        print(f"⚠️ No data for subclass {subclass}, skip plot.")
        continue

    plt.figure(figsize=(6, 4))
    sns.kdeplot(greedy_vals, fill=True, label="GAEC")
    sns.kdeplot(lookahead_vals, fill=True, label="Lookahead Greedy Joining")

    mean_g = np.mean(greedy_vals)
    mean_l = np.mean(lookahead_vals)

    plt.axvline(mean_g, color="blue", linestyle="--", label=f"GAEC mean: {mean_g:.2f}%")
    plt.axvline(mean_l, color="orange", linestyle="--", label=f"LGJ mean: {mean_l:.2f}%")

    plt.title(f"Normalized Cost Improvement: {subclass}")
    plt.xlabel("Improvement (%)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"class_{subclass}.png")
    plt.savefig(save_path)
    plt.close()

    summary_results.append({
        "Subclass": subclass,
        "GAEC_Mean(%)": mean_g,
        "LGJ_Mean(%)": mean_l,
        "Mean_Improvement(%)": mean_l - mean_g
    })

summary_df = pd.DataFrame(summary_results)
summary_csv_path = os.path.join(output_dir, "random_subclass_improvement_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

print(f"Summary saved: {summary_csv_path}")
print(summary_df.to_markdown(index=False, floatfmt=".2f"))