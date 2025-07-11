# calculating the mean_result for each category
import os
import pandas as pd
import numpy as np

csv_root = "result_csvs_unified_sparse"
output_dir = "mean_results_unified_sparse"
os.makedirs(output_dir, exist_ok=True)

round_to_integer = True 
decimal_precision = 6  

total_tasks = 0
completed_tasks = 0
results = []

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
        lookahead_csv = os.path.join(subdir_path, f"{instance_name}_greedy_joining_unified.csv")

        if not os.path.exists(lookahead_csv):
            print(f"Missing file for {instance_name}, skipping.")
            continue

        print(f"Processing {subdir}/{instance_name} ...")

        greedy_df = pd.read_csv(greedy_csv)
        lookahead_df = pd.read_csv(lookahead_csv)

        if round_to_integer:
            greedy_df["CleanCost"] = np.round(greedy_df["Cost"]).astype(int)
            lookahead_df["CleanCost"] = np.round(lookahead_df["Cost"]).astype(int)
        else:
            greedy_df["CleanCost"] = np.round(greedy_df["Cost"], decimals=decimal_precision)
            lookahead_df["CleanCost"] = np.round(lookahead_df["Cost"], decimals=decimal_precision)

        greedy_mean = greedy_df["CleanCost"].mean()
        lookahead_mean = lookahead_df["CleanCost"].mean()

        improvement = (lookahead_mean - greedy_mean) / greedy_mean * 100

        results.append({
            "Instance": f"{subdir}/{instance_name}",
            "GJ_Mean": greedy_mean,
            "Lookahead_Mean": lookahead_mean,
            "Improvement(%)": improvement
        })

        completed_tasks += 1
        print(f"Mean Cost → GJ: {greedy_mean:.2f}, Lookahead: {lookahead_mean:.2f}, "
              f"improvement: {improvement:.2f}%")

print(f"\nCompleted {completed_tasks}/{total_tasks} instances.\n")

results_df = pd.DataFrame(results)
results_csv_path = os.path.join(output_dir, "mean_comparison.csv")
results_df.to_csv(results_csv_path, index=False)

print(f"Summary CSV saved to {results_csv_path}\n")

print("### Improvement Summary\n")
print(results_df.to_markdown(index=False, floatfmt=".2f"))