# run the algorithm for all the dataset instances and save the results to csv for analysis
from greedy_joining import greedy_joining, greedy_joining_lookahead, greedy_joining_cohesion, greedy_joining_uniform
from time import time
from statistics import mean
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_rel
import csv


def load_cplib(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    weights = []
    for line in lines[1:]:
        weights += [float(x) for x in line.strip().split()]
    edges = []
    edge_weights = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = weights[idx]
            if w != 0.0:
                edges.append([i, j])
                edge_weights.append(w)
            idx += 1
    return n, edges, edge_weights


def load_cremi(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if lines[0].strip().upper() != "MULTICUT":
        raise ValueError("Not a valid MULTICUT file")
    edges = []
    weights = []
    node_ids = set()
    for line in lines[1:]:
        if line.strip() == "":
            continue
        i, j, w = line.strip().split()
        i, j, w = int(i), int(j), float(w)
        edges.append([i, j])
        weights.append(w)
        node_ids.update([i, j])
    n = max(node_ids) + 1
    return n, edges, weights


def load_graph_dataset(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
    if first_line.isdigit():
        return load_cplib(file_path)
    elif first_line.upper() == "MULTICUT":
        return load_cremi(file_path)
    else:
        raise ValueError(f"Unknown file format: {first_line}")


root_dir = "ABR_test"
runs = 20
csv_dir = "result_csvs_abr_test"
os.makedirs(csv_dir, exist_ok=True)

for subfolder in sorted(os.listdir(root_dir)):
    subfolder_path = os.path.join(root_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    print(f"Processing folder: {subfolder}")

    for fname in sorted(os.listdir(subfolder_path)):
        if not fname.endswith(".txt"):
            continue

        file_path = os.path.join(subfolder_path, fname)
        print(f"\nRunning on {subfolder}/{fname}")
        try:
            n, edges, weights = load_graph_dataset(file_path)
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
            continue

        for algo_name, algo_fn in [
            ("greedy_joining", greedy_joining),
            ("greedy_joining_lookahead", greedy_joining_cohesion)
        ]:
            print(f"   → {algo_name} ... ", end="", flush=True)
            costs = []
            for _ in range(runs):
                try:
                    cost, _ = algo_fn(n, edges, weights)
                    costs.append(cost)
                except Exception as e:
                    print(f"\nFailed run: {e}")
            print(f"Done")

            csv_subdir = os.path.join(csv_dir, subfolder)
            os.makedirs(csv_subdir, exist_ok=True)
            csv_file = os.path.join(csv_subdir, f"{fname.replace('.txt', '')}_{algo_name}.csv")
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Run", "Cost"])
                for i, cost in enumerate(costs):
                    writer.writerow([i + 1, cost])