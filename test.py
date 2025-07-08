from greedy_joining import greedy_joining
from greedy_joining import greedy_joining_lookahead
from greedy_joining import greedy_joining_cohesion
from greedy_joining import greedy_joining_uniform
from time import time
from statistics import mean
import os
from time import time
from statistics import mean
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

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
    

n, edges, weights = load_graph_dataset("bridges.txt")

costs = []
num_clusters = []
times = []

t_0 = time()
cut_value, cluster_labels = greedy_joining_cohesion(n, edges, weights)
t_1 = time()


print("Total cost:", cut_value)
print("Number of clusters:", len(set(cluster_labels)))
print("Time: ", t_1 - t_0)