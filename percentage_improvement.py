import pandas as pd

df = pd.read_csv("mean_results_unified_sparse/mean_comparison.csv")

df["Category"] = df["Instance"].apply(lambda x: x.split("/")[0])

grouped = df.groupby("Category").agg(
    Mean_Instance_Improvement=("Improvement(%)", "mean")
).reset_index()

print("\nCategory Mean Instance Improvement\n")
try:
    print(grouped.to_markdown(index=False, floatfmt=".2f"))
except ImportError:
    print(grouped.to_string(index=False, float_format="%.2f"))

grouped.to_csv("mean_results_unified_sparse/category_mean_instance_improvement.csv", index=False)
print("\nSummary saved to mean_results/category_mean_instance_improvement.csv")