# Distinguishing Physical and Cyber-Induced Forced-Oscillation Signatures Using Quantum Kernels Under Spatial Distribution Shift

Code, generated data, and result files for the IEEE Transactions on Consumer Electronics
manuscript TCE-2025-10-3616 (M. Suhail, A. Iqbal, M. Ullah, M. N. Aman, B. Sikdar).

## Layout

| Folder | Contents |
|---|---|
| `quantum_kernel/` | `qsvm_kernel_v2.py` — fixed fidelity-kernel feature map (Qiskit statevector), Gram normalization/centering (Eq. 15), CV selection. `qsvm_kernel_v2_numpy.py` — Qiskit-free drop-in giving identical kernels (verified to ~1e-16; every reported AUC reproduces exactly). |
| `synthetic/` | MATLAB generator of the 12-PMU shifted-footprint data (`run_stageB_prime_export.m`, `run_stageB_prime_all_seeds.m`), the Python comparison (`run_stageB_prime.py`), the 10-D topology ablation (`run_stageB_ablation_spoof.m`), and `data/stageB_prime_confirmatory/` — the generated feature tables and per-seed results for the frozen seeds 100–119. |
| `isone/` | ISO-NE event-held-out validation with paired semi-synthetic timing attacks (descriptor construction, filtering, interpolation, missing-value handling, PCA, nested event-grouped selection) and their result JSON files. |
| `revision_round2/` | Analyses added in the second revision: finite-shot / depolarizing-noise emulation and cost (`realism_frozen.py`), FakeTorino transpilation and Aer noise check (`transpile_noise_check.py`), additional classical baselines, topology ablation on the headline representation and full-training-set runs (`synth_r4.py`, `extra_models.py`, `grid_ext.py`), aggregate paired ISO-NE statistics (`isone_repro.py`, `agg_ci.py`, `agg_run.py`), and ISO-NE robustness controls (`attack_variants.py`, `isone_variants.py`). All outputs are in `revision_round2/results/`. |
| `figures/` | `make_revision_figures.m` — regenerates the manuscript figures from the result files. |

## Environment

Python 3.11, Qiskit 2.1.1, scikit-learn 1.7.1, NumPy 2.3.2, SciPy, pandas (see `requirements.txt`);
`qiskit-aer` and `qiskit-ibm-runtime` only for `transpile_noise_check.py`; MATLAB R2023b or later
(Signal Processing Toolbox) for data generation and figures. Reported runs used an Apple M4 Max
(14 cores, 36 GB, macOS 15.7.3); revision-round analyses were also run on Linux x86-64.

## ISO-NE data

The six ISO-NE records are **not redistributed**. Download them from the IEEE-NASPI
*Test Cases Library on Forced/Sustained Power System Oscillations*
(https://web.eecs.utk.edu/~kaisun/Oscillation/, IEEE DataPort doi:10.21227/a6hg-n822),
save them as `ISO-NE_case1.csv` … `ISO-NE_case6.csv`, and verify them against `ISO-NE_sha256.txt`.

## Reproducing the results

```bash
export PYTHONPATH=$PWD/quantum_kernel:$PWD/isone:$PWD/revision_round2
```

| Manuscript item | Command | Output |
|---|---|---|
| Synthetic data, seeds 100–119 | MATLAB: `run_stageB_prime_all_seeds('Seeds',100:119,'MaxTrain',400)` | `synthetic/data/stageB_prime_confirmatory/` (already included) |
| Fig. 5, frozen 20-seed comparison | produced by the command above | `stageB_prime_confirmatory_summary.csv` |
| Topology ablation, 10-D (Sec. VI-A) | MATLAB: `run_stageB_ablation_spoof` | `stageB_ablation_results.mat` |
| ISO-NE Table II, Figs. 6–7 | `python isone/run_isone_final_matched.py --data <isone dir> --qsvm quantum_kernel/qsvm_kernel_v2.py` | `isone/results/isone_final_matched.json` |
| Representation ablation (Fig. 8) | `python isone/run_isone_nonstationary_experiments.py --data <isone dir> --qsvm quantum_kernel/qsvm_kernel_v2.py` | `isone/results/isone_nonstationary.json` |
| Case 6 check | `python isone/run_isone_semisynthetic_validation.py --data <isone dir> --qsvm quantum_kernel/qsvm_kernel_v2.py` | `isone/results/isone_validation.json` |
| Table II aggregate CIs | from a folder holding the ISO-NE CSVs: `python revision_round2/isone_repro.py --out repro_published.json`, then `python revision_round2/agg_run.py repro_published.json agg_published.json QSVM-RBF,QSVM-LR,RBF-LR` | `results/agg_published.json` |
| Tables III–IV (cost, shots, noise) | `python revision_round2/realism_frozen.py synthetic/data/stageB_prime_confirmatory <ref.json> realism_frozen.json` | `results/realism_frozen.json` |
| FakeTorino transpilation / Aer check | `python revision_round2/transpile_noise_check.py` (run inside the folder holding `stageB_prime_confirmatory/` and `qsvm_kernel_v2.py`) | `isone/results/transpile_noise_check.json` |
| Additional baselines | `TCE_SYNTH_DATA=synthetic/data/stageB_prime_confirmatory python revision_round2/synth_r4.py synthetic/data/stageB_prime_confirmatory baselines 100,...,119 base.json`; widened grids: `python revision_round2/grid_ext.py 100,...,119 grid_ext.json`; ISO-NE: `python revision_round2/isone_repro.py --extra Poly-SVM,Laplacian-SVM,MLP,GBoost,RFF-256,CosProd-SVM --out isone_extra.json` | `results/` |
| Topology on headline representation | `... synth_r4.py <data> topology 100,...,119 topo.json` | `results/topo_*.json` |
| Full 2000-window training | `... synth_r4.py <data> full 100,...,119 full.json` | `results/full_all.json` |
| Robustness controls (Table VI) | from a folder holding the ISO-NE CSVs: `python revision_round2/isone_variants.py 400,401,402,403,404 variants.json` | `results/variants.json`, `results/agg_var_*.json` |

Every script fixes its random seeds; `realism_frozen.py`, `isone_repro.py`, and the topology run
check their reproduced AUCs against the archived values (all differences are exactly zero).

## License

Code: MIT (see `LICENSE`). Generated synthetic data and result files: CC BY 4.0.
