# Research and Experiments for Variants of the Greedy Joining Algorithm

This repository contains implementations and experiments for several variants of the classic Greedy Joining algorithm. The goal of this project is to explore structural extensions and regularization mechanisms that improve clustering quality on signed graph instances.

## Acknowledgement

This project incorporates parts of the [bjoern-andres/graph](https://github.com/bjoern-andres/graph) library developed by Bjoern Andres, including selected components for graph data structures and Python-C++ interface bindings.

The original library is licensed under the BSD 3-Clause License (see [LICENSE](./LICENSE)) and is further described on the author's [official website](http://www.andres.sc/graph.html). We retain the original license notice and acknowledge its contributions to the interface and structural foundation used in our algorithmic extensions.

All newly developed algorithmic components in this repository are original work and are licensed separately (see [License](#license) below).

## File Structure

- `test.py`  
  Run a single experiment on a specific instance.

- `alg_all.py`  
  Run a specified algorithm multiple times and save results to a CSV file.

- `mean_results.py`  
  Compute average results for a given algorithm.

- `percentage_improvement.py`  
  Calculate the improvement over the original Greedy Joining algorithm.

- `plot_single_op.py`  
  Plot performance results for each instance.

- `distribution_category.py`  
  Visualize improvement distributions across instance categories.

## Algorithm Variants

Located in `greedy-joining/include/`:

- `greedy-joining-lookahead.hxx`  
  Lookahead variant using structural conflict regularization.

- `greedy-joining-cohesion.hxx`  
  Variant using structural cohesion (positive reinforcement).

- `greedy-joining-uniform.hxx`  
  Uniform strategy without structure-based priorities.

## Notes

- Results are saved as CSV files for further statistical and visual analysis.
- Designed for experimentation on the CP-Lib benchmark instances.

## License

This repository contains code under multiple licenses:

- Components derived from the `graph` library by Bjoern Andres are licensed under the BSD 3-Clause License. We retain the original license text and attribution in the LICENSE file.
- All other code developed for this research (especially files under `greedy-joining/include/` and Python scripts) is released under the MIT License.

See the [LICENSE](./LICENSE) file for full details.
