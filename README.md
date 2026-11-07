# RFTIC Insight Model

Reproducible code, data, and numerical validation for the reduced stochastic model presented in *The Architecture of Insight: A Reduced-Model Hypothesis for Representational Flexibility and Insight Capacity*.

## What this repository contains

- [Manuscript](paper/main.pdf), [Supplementary Information](paper/supplementary.pdf), and their [LaTeX sources](paper/).
- [Production summary with corrected confidence intervals](data/final_sweep_data_CORRECTED_CI.csv), [fixed seed registry](data/final_sweep_seeds.json), and [per-trajectory escape-time data](data/raw/).
- [Simulation and statistical safeguards](scripts/), including the one-dimensional MFPT quadrature, the Kramers high-barrier reference, and the Figure 1 plotting code.
- [Figure 1](figures/figure1_final_validation.png), [symbolic and numerical reduction checks](validation/reduction_audit_checks_output.txt), and the [SHA-256 manifest](MANIFEST.md).

## Scientific scope

This package checks mathematical and computational consistency within the canonical one-dimensional stochastic normal form. Human representational flexibility and its clinical, neural, and artificial-intelligence interpretations remain hypotheses outside the scope of these checks.

## Production sweep

The sweep uses 16 log-spaced values of the barrier-to-noise ratio `r = ΔE/T` from 0.15 to 7, with 6,000 trajectories per condition and 96,000 in total. The canonical time step is 0.01 and the right-censoring horizon is 150. Trajectories start at the negative well minimum. The event is the first arrival at the positive well minimum, the committed-transition threshold.

The fit uses total time at risk with right censoring. An empirical hazard-stabilization rule selects the left-truncation time. The log-normal confidence interval uses the post-truncation event count and conditions on the selected time. The stabilized conditional hazard and the inverse point-start MFPT are distinct estimands at low barriers. The Kramers curve is the high-barrier asymptotic reference.

The supplied NPZs were deterministically reconstructed from the archived seeds using the production implementation and matched the archived event counts, truncation times, and stabilized estimates. Each NPZ stores the scalar condition metadata, 6,000 canonical escape times, and 6,000 Boolean event flags. Censored trajectories have a NaN escape time and contribute the stored censoring horizon. The reconstruction also stored the selected truncation time and the event counts before and after truncation. Its recorded environment was Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, and pandas 3.0.1.

## Reproduce from the archived data

Use 64-bit Python 3.12. The direct dependencies are pinned in [requirements.txt](requirements.txt).

```bash
git clone https://github.com/Khaos41x/rftic-insight-model.git
cd rftic-insight-model
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/reproduce.py
```

On Linux or macOS:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/reproduce.py
```

The quick check validates the 16 NPZ payloads and the seed registry, then recalculates the event counts, selected truncation times, rates, corrected intervals, and plotted comparison metrics. It exits with PASS or FAIL and writes the regenerated plot to `build/figure1_final_validation.png` after the numerical checks pass. The packaged Figure 1 and input data stay in place. Rendering may vary with the platform and plotting dependencies.

To regenerate just the plot from the summary CSV, run `python scripts/plot_final_validation.py` with the virtual environment active. To rerun the mathematical sanity checks, run `python scripts/reduction_audit_checks.py` and compare with the [archived output](validation/reduction_audit_checks_output.txt). The Euler–Maruyama step-size helper is in [safeguards.py](scripts/safeguards.py). The standalone per-step output from the original lower-barrier checks is unavailable.

## Optional full simulation

With the virtual environment active, run these separately from the fast workflow:

```bash
python scripts/final_sweep.py
python scripts/reproduce.py --raw-dir build/full-simulation/data/raw_data
```

The full runner uses the production grid and seeds. It requires a fresh `build/full-simulation/` directory and writes its outputs there. The second command applies the same post-truncation checks to the new raw files and compares them with the shipped corrected summary. Cross-platform floating-point or random-generator differences can affect the deterministic rerun.

## Build the manuscript

From the repository root, with a TeX distribution that provides pdfLaTeX and BibTeX:

```bash
cd paper
pdflatex -halt-on-error main.tex
bibtex main
pdflatex -halt-on-error main.tex
pdflatex -halt-on-error main.tex
pdflatex -halt-on-error supplementary.tex
pdflatex -halt-on-error supplementary.tex
```

## Citation and license

If you use the software or data, please cite the manuscript and this package. See [CITATION.cff](CITATION.cff) and the [MIT license](LICENSE).
