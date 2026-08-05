# Numerical experiments

This directory contains the numerical-experiments package for the
paper:

**Exact Finite-Horizon Memory, Conditioning, and Dissipative Decay in
Coarse Upwind Finite-Volume Prediction**

The manuscript is in final preparation for submission to the
**Journal of Scientific Computing**; its sources and PDF will appear
in the adjacent `manuscript/` directory. These experiments produce
the figures, CSV data, and LaTeX tables used in the paper. They are
floating-point illustrations of the paper's exact results; the exact
verification lives in the adjacent `certificates/` directory, which
replays the underlying identities in exact arithmetic.

## Overview

The paper studies a simple question in coarse finite-volume prediction:

> If we know only coarse parent-cell averages, what additional information is needed to predict the future parent averages of the same fine-grid realization?

The analysis is carried out for one-dimensional periodic scalar advection discretized by a first-order upwind finite-volume method with forward Euler time integration.

The paper separates three concepts:

1. **Exact memory:** the additional real-valued information needed for exact pathwise coarse prediction.
2. **Stable observability:** whether the algebraically visible information is recoverable above a declared singular-value tolerance.
3. **Dissipative decay:** whether unresolved nonconstant information decays under the numerical dissipation of the upwind method.

The code in this repository reproduces the numerical figures and tables used in the paper.

## Contents

```text
.
├── run_experiments.py        # Runs all numerical experiments and writes figures/CSV files
├── make_tables.py            # Converts CSV outputs into LaTeX table fragments
├── LICENSE                   # MIT License (this directory)
└── README.md
```

After running the scripts, this directory will contain the following generated files.

### Figures

```text
figures/rank_effective.pdf
figures/singular_values_saturated.pdf
figures/queue_conditioning.pdf
figures/step_flattening.pdf
figures/delayed_collision_erasure.pdf
```

### CSV Data

```text
results/rank_effective.csv
results/queue_conditioning.csv
results/step_flattening.csv
results/delayed_collision_erasure.csv
results/flux_closure_check.csv
```

### LaTeX Tables

```text
tables/rank_saturation_table.tex
tables/step_final_table.tex
```

## Mathematical Model

The experiments use the periodic scalar upwind update

$$
u_i^{n+1} = (1-\lambda)u_i^n + \lambda u_{i-1}^n,
$$

with periodic indexing.

The total number of fine cells is

$$
N = P r,
$$

where:

- $P$ is the number of parent cells;
- $r$ is the number of fine cells per parent;
- $N$ is the total number of fine-grid cells.

The parent-average restriction matrix is denoted by $R$, and the one-step upwind matrix is denoted by $A_\lambda$.

The finite-horizon observation matrix is

$$
\mathcal O_L =
\begin{bmatrix}
R \\
R A_\lambda \\
\vdots \\
R A_\lambda^L
\end{bmatrix}.
$$

The paper proves the exact rank formula

$$
rank \mathcal O_L = P + (P-1)\min(L,r-1).
$$

The numerical experiments illustrate this exact rank formula, the effective-rank loss caused by conditioning, and dissipative decay of unresolved information.

## Experiments

### 1. Exact Rank Versus Effective Rank

Implemented in:

```text
experiment_rank_effective()
```

This experiment constructs the observation matrix $\mathcal O_L$ and compares:

- the exact rank formula;
- the effective numerical rank computed from singular values.

Parameters used in the paper:

```text
P = 8
r = 6
N = 48
lambda in {1, 0.5, 0.1, 0.01, 0.001}
L = 0, ..., r + 3
```

Outputs:

```text
figures/rank_effective.pdf
results/rank_effective.csv
```

It also generates the saturated singular-value figure:

```text
figures/singular_values_saturated.pdf
```

### 2. Saturated Singular Values

Also implemented in:

```text
experiment_rank_effective()
```

This experiment fixes the saturated horizon

$$
L = r - 1
$$

and plots all singular values of the saturated observation matrix.

For the paper parameters, $P = 8$, $r = 6$, and $L = 5$, so

$$
\mathcal O_5 \in \mathbb R^{48 \times 48}.
$$

The exact saturated rank is $43$, so five singular values vanish in exact arithmetic.

Output:

```text
figures/singular_values_saturated.pdf
```

### 3. Queue Conditioning

Implemented in:

```text
experiment_queue_conditioning()
```

This experiment constructs the collar-to-queue matrix $T_q$, which maps right-collar subcell values to future outflow entries.

The paper proves that the smallest singular value of this matrix scales like

$$
\lambda^q
$$

for small $\lambda$.

Parameters used in the paper:

```text
q = 1, ..., 5
lambda logarithmically spaced from 1e-4 to 1
```

Outputs:

```text
figures/queue_conditioning.pdf
results/queue_conditioning.csv
```

Note: the script uses double precision. The analytical theorem in the paper establishes the small-$\lambda$ scaling exactly; very small singular values in the figure should be interpreted as a visual diagnostic rather than as high-relative-accuracy measurements.

### 4. Periodic Step-Function Flattening

Implemented in:

```text
experiment_step_flattening()
```

This experiment evolves the periodic step function

$$
u_i^0 =
\begin{cases}
1, & 0 \le i < N/2, \\
0, & N/2 \le i < N.
\end{cases}
$$

The following diagnostics are recorded:

$$
E_2(n) = \|u^n - \bar u^n \mathbf 1\|_2,
$$

$$
E_\infty(n) = \|u^n - \bar u^n \mathbf 1\|_\infty,
$$

$$
TV(n) = \sum_i |u_i^n - u_{i-1}^n|,
$$

and

$$
M(n) = |\bar u^n - \bar u^0|.
$$

Parameters used in the paper:

```text
N = 64
nsteps = 60000
lambda in {0.5, 0.1, 1.0}
```

Outputs:

```text
figures/step_flattening.pdf
results/step_flattening.csv
```

For $0 < \lambda < 1$, the upwind scheme is dissipative and the nonconstant component decays. For $\lambda = 1$, the method is a cyclic shift and the step does not flatten.

### 5. Delayed Collision and Dissipative Decay

Implemented in:

```text
experiment_delayed_collision_erasure()
```

This experiment constructs two fine-grid states $x^0$ and $y^0$ with the same parent averages:

$$
R x^0 = R y^0.
$$

However, the two states differ inside one parent cell by a compensated subcell perturbation. The perturbation is initially hidden from the parent averages and becomes visible only after a delay.

The diagnostic is

$$
D(n) = \|R A_\lambda^n x^0 - R A_\lambda^n y^0\|_2.
$$

Parameters used in the paper:

```text
P = 8
r = 8
N = 64
lambda = 0.5
delay_m = 3
perturbation amplitude = 0.25
nsteps = 60000
```

Outputs:

```text
figures/delayed_collision_erasure.pdf
results/delayed_collision_erasure.csv
```

This experiment illustrates the full mechanism:

1. two fine states initially have identical parent averages;
2. their parent averages separate after hidden subcell information reaches a parent interface;
3. the difference later decays because the upwind scheme is dissipative.

### 6. One-Step Flux-Closure Verification

Implemented in:

```text
experiment_flux_closure_check()
```

This check verifies the telescoping finite-volume identity used in the paper.

For the scalar upwind scheme, the realized parent-interface flux is

$$
\Phi_K = \lambda x_{K,r-1}.
$$

The check compares:

1. update the fine solution and then restrict to parent averages;
2. update the parent averages using the realized flux difference.

Parameters used in the paper:

```text
P = 10
r = 4
N = 40
lambda = 0.37
ntrials = 100
random seed = 12345
random distribution = standard normal
```

Output:

```text
results/flux_closure_check.csv
```

## Installation

Create a Python environment and install the required packages.

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy matplotlib
```

## Running the Experiments

From the repository root, run:

```bash
python run_experiments.py
```

This creates the `figures/` and `results/` directories and writes all figures and CSV files.

Then generate the LaTeX table fragments:

```bash
python make_tables.py
```

This creates the `tables/` directory and writes:

```text
tables/rank_saturation_table.tex
tables/step_final_table.tex
```

## Reproducing the Paper Figures and Tables

A complete reproduction sequence is:

```bash
python run_experiments.py
python make_tables.py
```

Then compile the manuscript in LaTeX. The manuscript expects the figure files in `figures/` and may include the generated table fragments from `tables/`.

## Expected Outputs

After running both scripts, you should have:

```text
figures/rank_effective.pdf
figures/singular_values_saturated.pdf
figures/queue_conditioning.pdf
figures/step_flattening.pdf
figures/delayed_collision_erasure.pdf

results/rank_effective.csv
results/queue_conditioning.csv
results/step_flattening.csv
results/delayed_collision_erasure.csv
results/flux_closure_check.csv

tables/rank_saturation_table.tex
tables/step_final_table.tex
```

The terminal will also print a summary of the one-step flux-closure check, including the maximum and mean discrepancy over the random trials.

## Numerical Precision

The scripts use double-precision NumPy arrays.

The effective rank is computed from the singular values of the observation matrix using the tolerance

$$
\tau =
100 \epsilon_{\rm mach}
\max\{(L+1)P, Pr\}
\|\mathcal O_L\|_2.
$$

This is a declared numerical diagnostic. It is not a new mathematical rank.

The queue-conditioning figure includes very small singular values. The paper's analytical theorem establishes the scaling; very small double-precision singular values should be interpreted as visual diagnostics rather than high-relative-accuracy measurements.

## Randomness and Determinism

All experiments are deterministic except for the one-step flux-closure verification check.

That check uses:

```python
np.random.default_rng(12345)
```

with standard normal random fine-grid states.

Therefore, the random check is reproducible.



## License

The source code in this repository is licensed under the **MIT License** (see `LICENSE`)


## Citation

A citation entry will be added after the manuscript is accepted or assigned a preprint DOI.

For now, please cite the manuscript as:

```text
Polemitis, A., Christakis, N., and Drikakis, D. Exact Finite-Horizon Memory, Conditioning, and Dissipative Decay in Coarse Upwind Finite-Volume Prediction. Manuscript submitted to Journal of Scientific Computing.
```


```
