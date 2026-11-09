# Global convexification of neural network residual learning

Code for the numerical examples of the paper *Global Convexification of Neural Network Residual Learning*.

A single-hidden-layer tanh network g(x; Θ) learns the residual r_n = x_{n+1} − f̂(x_n) of a nominal model f̂ of the Hénon map. The regulariser

R(Θ; r) = μ/2 ‖Θ‖² + γ/4 ‖Θ‖⁴ − E_r v₀ᵀΘ

makes J = L₀ + R globally strongly convex for μ > K₀ and γ ≥ K₂. Two-phase training minimises J until a gradient-based release condition holds and then minimises L₀.

## Requirements

- MATLAB R2020b or later. The Parallel Computing Toolbox is optional (`parfor` runs serially without it).
- For Figure 2 only: Python 3 with NumPy, SciPy and Matplotlib.

## Reproducing the results

Run from this folder. All output goes to `results/`.

| Script | Paper | Run time |
|---|---|---|
| `run_perfect.m` | Figures 1 and 3, Section 4.1 | about 1 min |
| `run_imperfect.m` | Figure 4, Table 2, Section 4.2 | about 1.5 h (N = 400: about 1 h) |
| `python3 fig_two_phase.py` | Figure 2 | a few seconds |

Run times are for a laptop with the Parallel Computing Toolbox.

## Files

| File | Content |
|---|---|
| `henon_data.m` | Hénon-map samples and nominal-model residuals |
| `objective.m` | L₀, J and the gradient |
| `reg_constants.m` | K₀ and K₂ (Proposition 1) |
| `v0_direction.m` | v₀ from the residual–input correlation C (Proposition 2) |
| `make_regulariser.m` | μ = 1.01 K₀, γ = K₂, E_r v₀ |
| `train_run.m` | one Adam run: `'L0'`, `'J'` or `'two-phase'` (Algorithm 1) |
| `default_options.m` | Adam and step-size settings |
| `init_params.m` | random initialisation Θ₀ ~ N(0, I) |
| `plot_perfect.m`, `plot_imperfect.m`, `plot_band.m`, `save_fig.m` | figures |
| `table_imperfect.m` | Table 2 |

## Parameter layout

Θ = [W₂(:); W₁(:); b₁] with output weights W₂ (N × 2), input weights W₁ (N × 2) and biases b₁ (N × 1), so that g(x; Θ) = W₂ᵀ tanh(W₁x + b₁). The paper orders the parameters neuron by neuron; the norms and all results are the same.
