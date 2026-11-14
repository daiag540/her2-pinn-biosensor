# HER2 Aptasensor — PINN Training Code and Data

**Corresponding manuscript:** *A Dual-Aptamer Y-Shaped DNA Molecular-Pendulum Biosensor for HER2 with Physics-Informed Optimization of AuNP Electrodeposition* (Anguo Dai, Zhuofan Chen, Xingliang Xiong; target journal: *Sensors and Actuators B: Chemical*).

**Paper DOI:** *to be added after publication* · **Archive DOI (Zenodo):** *to be added after the v1.0.0 release is linked to Zenodo*

This repository contains **only the physics-informed neural network (PINN) training code, its input data and its outputs**. It does not contain manuscript figures, Illustrator/JSX drawing scripts, Word documents or image assets. The PINN is an **optimisation tool** constrained by modified Cottrell transport and impedance evolution; every modelling claim in the manuscript is backed by an electrochemical measurement. It is not a purely theoretical study.

Training locates an apparent cumulative percolation transition at *t*<sub>crit</sub> = 16.18 s (*Q*<sub>c</sub> = 0.748 mC, *D*<sub>eff</sub> ≈ 2.69 × 10⁻⁴ cm² s⁻¹); the resulting deposition protocol uses 3.00 s unit pulses over four cycles (12.00 s total active deposition). The associated analytical figures of merit (LOD 4.752 pg mL⁻¹ over a tested 5–180,000 pg mL⁻¹ range in undiluted human serum) are reported in the manuscript, not reproduced here. This is an analytical proof-of-concept artefact, not a clinical validation dataset.

---

## 1. Directory structure

```
her2-pinn-biosensor/
├── README.md              # this file: overview, structure, run instructions, data dictionary, citation
├── LICENSE                # MIT License for the code
├── CITATION.cff           # machine-readable citation (GitHub "Cite this repository")
├── requirements.txt       # pinned core library versions
├── .gitignore             # excludes caches / IDE / OS files
├── MANIFEST.md            # provenance map: every archived file -> original source + sha256
├── data/
│   ├── LICENSE            # CC BY 4.0 for the experimental data
│   ├── raw/               # 30 unmodified CHI604E exports (read-only archive)
│   │   ├── it_1.txt  … it_10.txt
│   │   ├── ocp_1.txt … ocp_10.txt
│   │   └── eis_1.txt … eis_10.txt
│   └── processed/         # standard input table
│       └── pinn_final_training_set.csv
├── scripts/
│   └── pinn_workflow.py   # extraction + PINN training + transition analysis + sensitivity + Figure S1
└── results/               # outputs of the archived training run
    ├── biosensor_pinn_model.pth, biosensor_pinn_model_best.pth
    ├── pinn_loss_history.csv, training_history.json
    ├── pinn_sensitivity_results.csv, pinn_training.log
    ├── pinn_pierce_point_result.png
    └── Figure_S1_PINN_Training_and_Sensitivity.png / .pdf
```

`MANIFEST.md` records where each file came from (paths relative to the manuscript folder, never a local absolute path) and a short SHA-256 digest so the archived copy can be verified against the original. Deliberately excluded: all `.jsx` / `.ai` drawing sources and their raster assets, manuscript `Figure_1.pdf` … `Figure_14.pdf`, the manuscript and cover letter, and an outdated equation sheet whose 3.00 s / 807.73 Ω values were superseded by the final 16.18 s / 754.63 Ω values.

---

## 2. Environment and reproduction

Tested with **Python 3.12.7** on Windows; the code is OS-independent (all paths resolve relative to the repository root).

```bash
git clone https://github.com/daiag540/her2-pinn-biosensor.git
cd her2-pinn-biosensor
python -m pip install -r requirements.txt
python scripts/pinn_workflow.py
```

`requirements.txt` pins `torch 2.5.1`, `numpy 2.0.0`, `pandas 2.3.3`, `scipy 1.16.0` and `matplotlib 3.10.8` — the versions used to produce the archived results.

What the script does, in order:

1. **Data extraction** — skipped when `data/processed/pinn_final_training_set.csv` exists; otherwise it reads `data/raw/it_*.txt`, `ocp_*.txt`, `eis_*.txt` and rebuilds that table into `data/processed/`.
2. **PINN training** — 5001 epochs, `hidden_width=64`, `lr=1e-3`, `lambda_phys=10.0`, `lambda_z=1e-3`, recording all loss components every 10 epochs; writes the loss CSV/JSON and both model checkpoints into `results/`.
3. **Transition-point analysis** — extracts *t*<sub>crit</sub>, *Q*<sub>c</sub>, *R*<sub>c</sub>, *D*<sub>eff</sub> and the impedance derivative.
4. **Sensitivity study** — sweeps λ<sub>phys</sub>, λ<sub>Z</sub>, network width and learning rate.
5. **Figure S1** — exports the four-panel training/sensitivity figure as 600 dpi PNG and vector PDF into `results/`.

To point the script at a different data folder, set `PINN_DATA_DIR` (it must contain `raw/` and `processed/`):

```bash
PINN_DATA_DIR=/path/to/other_data python scripts/pinn_workflow.py
```

The full run retrains the main model and four sensitivity sweeps of 5001 epochs each; a CUDA-capable GPU is recommended. The archived `results/` files let readers inspect all outputs without rerunning.

**Reproducibility:** `set_seed(42)` fixes NumPy, Python `random` and CPU/CUDA Torch seeds at the start of training; the main workflow and the sensitivity study both pass `seed=42`. No absolute local paths appear in any script.

### 2.1 Physics constraints implemented in `pinn_workflow.py`

Inputs are Z-score normalised, and the network predicts `I_norm`, `R_norm` and `OCP_norm` from `t`. The total loss is

```latex
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}}\mathcal{L}_{\text{Cottrell}} + \mathcal{L}_{\text{phase}} + \lambda_z \mathcal{L}_{\text{deriv}}
```

```latex
|I_{\text{theory}}(t)| = \frac{n F A_{\text{eff}} 10^{\frac{1}{2}\log_{10}D_{\text{eff}}} C}{\pi^{1/2}(t + |\delta t|)^{1/2}}, \qquad I_{\text{obs,approx}} = \frac{Q}{\Delta t}\times 10^{-3}\ \text{A}
```

```latex
\mathcal{L}_{\text{phase}} = \frac{1}{N}\sum_i \frac{\left(R_{\text{pred,real}}(t_i) - R_{\text{phase,theory}}(Q_i)\right)^2}{\sigma_R^2}, \qquad
R_{\text{phase,theory}}(Q) = R_{\text{plateau}} + \frac{R_{\text{initial}} - R_{\text{plateau}}}{1 + \exp\left(-\frac{Q - Q_c}{0.05}\right)}
```

with `n = 3`, `F = 96,485 C mol⁻¹`, `C = 1.0 × 10⁻⁶ mol cm⁻³`, `A_base = 0.1963 cm²`, `σ = 1.6688`, `λ_phys = 10.0`, `λ_z = 1.0 × 10⁻³`. The learned parameters are `log₁₀ D_eff`, `Q_c` and the singularity regulariser `|δt|` (initial 0.005 s). The transition is read from the impedance record as

```latex
t_{\text{crit}} = \arg\max_{t \in [3.0,\ 30.0]} \left| \frac{\partial Z_{\text{real}}}{\partial t} \right| = 16.18\ \text{s}
```

Gradient smoothing is penalised by `L_deriv = (10⁻⁴/N) Σ |∂R_pred,real/∂t|`. The archived sensitivity CSV shows *t*<sub>crit</sub> = 16.1826 s and *Q*<sub>c</sub> = 0.7484 mC are stable across all λ<sub>phys</sub> and λ<sub>Z</sub> sweeps.

---

## 3. Data dictionary

### 3.1 `data/raw/` — instrument exports (one trio per deposition segment, 1–10)

| File pattern | Technique | Columns / meaning | Conditions recorded in header |
|---|---|---|---|
| `it_<n>.txt` | Chronoamperometry | `Time/sec`, `Current/A`; `\|I\|` is integrated with `scipy.integrate.trapezoid` over time to give segment charge (mC) | Init E 0.8 V, High E 0.548 V, Low E 0 V, Pulse Width 2 s, Sample Interval 1e-3 s |
| `ocp_<n>.txt` | Open circuit potential – time | `Time/sec`, `Potential/V`; second column averaged per segment | Run Time 3 s |
| `eis_<n>.txt` | Impedance – potential | `Potential/V`, `Z'/ohm`, `Z"/ohm`, `Z/ohm`, `Phase/deg`; **the second column `Z'` is averaged** to give `Z_real` | Init E 0.548 V → 0.568 V, Incr 0.01 V, 10 Hz, amplitude 0.005 V, quiet time 2 s |

All 30 files are unmodified exports from a CHI604E potentiostat. Headers were checked individually: they contain only a timestamp, instrument model, technique name and acquisition parameters — no computer user name, operator name or institutional identifier.

### 3.2 `data/processed/pinn_final_training_set.csv`

| Column | Physical meaning | Unit |
|---|---|---|
| `segment` | Index of the 3.00 s unit deposition pulse (1–10) | dimensionless |
| `Time` | Cumulative active deposition time (3.0 … 30.0) | s |
| `Q_mC` | Cumulative charge from the trapezoidal integration of the `it` curve | mC |
| `OCP` | Mean open-circuit potential of the segment | V |
| `Z_real` | Mean real-part impedance `Z'` of the segment | Ω |
| `sigma` | Fixed dimensionless transport/geometry constant `STATIC_PARAMS['sigma']` = 1.6688 | dimensionless |
| `E_onset` | Fixed onset potential `STATIC_PARAMS['E_onset']` = 0.5983 | V |
| `A_base` | Fixed working-electrode area (5 mm diameter disk) `STATIC_PARAMS['A_base']` = 0.1963 | cm² |
| `dR_dQ` | Impedance change per unit charge, dR/dQ | Ω mC⁻¹ |

### 3.3 `results/`

| File | Content |
|---|---|
| `pinn_loss_history.csv` | Every 10th epoch: `epoch, total_loss, data_loss, R_loss, OCP_loss, cottrell_loss, phase_loss, deriv_loss, D, Qc, delta_t` |
| `training_history.json` | Same loss trajectories as JSON lists |
| `pinn_sensitivity_results.csv` | `sweep_type, param_value, lambda_phys, lambda_z, hidden_width, lr, t_crit, D_eff, Q_c` for all four sweeps |
| `pinn_training.log` | Epoch-level logging output |
| `biosensor_pinn_model_best.pth` / `biosensor_pinn_model.pth` | Best / final Torch checkpoints (state dict + learned physics parameters) |
| `pinn_pierce_point_result.png` | Transition-node diagnostic plot |
| `Figure_S1_PINN_Training_and_Sensitivity.png` / `.pdf` | Four-panel Supporting Information figure (600 dpi PNG + vector PDF), produced directly by the workflow |

---

## 4. Citation

GitHub shows a **Cite this repository** button backed by `CITATION.cff`. BibTeX:

```bibtex
@misc{dai2026her2pinn,
  author       = {Dai, Anguo and Chen, Zhuofan and Xiong, Xingliang},
  title        = {HER2 Dual-Aptamer Y-Shaped DNA Molecular-Pendulum Biosensor: PINN Code and Data},
  year         = {2026},
  howpublished = {\url{https://github.com/daiag540/her2-pinn-biosensor}},
  note         = {Version 1.0.0; Zenodo DOI to be added after archive release}
}

@article{dai2026her2sensor,
  author  = {Dai, Anguo and Chen, Zhuofan and Xiong, Xingliang},
  title   = {A Dual-Aptamer Y-Shaped DNA Molecular-Pendulum Biosensor for HER2 with Physics-Informed Optimization of AuNP Electrodeposition},
  journal = {Sensors and Actuators B: Chemical},
  year    = {2026},
  note    = {Manuscript; add DOI after publication}
}
```

---

## 5. Licence

- **Code / scripts:** [MIT](LICENSE).
- **Experimental data (`data/`):** [CC BY 4.0](data/LICENSE) — reuse is permitted with attribution to the manuscript and this repository.
- The Figure S1 files in `results/` accompany the unpublished manuscript; please cite the article once available.

---

## 6. Platform limits, privacy and academic archiving

- **Size:** 48 files, ≈1.97 MB in total, largest single file ≈0.96 MB — well below GitHub's 100 MB per-file hard limit and the recommended 1 GB repository size; no Git LFS or external hosting needed.
- **De-identification:** raw headers contain no user/operator/institution identifiers; scripts contain no API keys, mailbox passwords, intranet addresses or hard-coded absolute paths. Commercial, de-identified serum was used in vitro only; no human-subject identifiers are present.
- **Permanent DOI:** a `v1.0.0` GitHub release freezes the archive. Link the repository from a Zenodo account (Zenodo → GitHub → enable this repository) to mint a DOI, then paste it into the manuscript's *Data Availability* statement and the placeholder at the top of this README. A DOI is preferable to a bare GitHub URL because repositories can be edited or deleted.
- **Peer-review compatibility:** *Sensors and Actuators B: Chemical* uses single-blind review, so a normal attributed repository is acceptable. If the manuscript is later submitted to a double-blind journal, serve an **Anonymous GitHub** mirror during review and ensure code comments and Git commit history carry no author identity.

---

## 7. Scope and limitations

The PINN here optimises an electrode-fabrication parameter; it is not a clinical diagnostic model. Serum experiments used commercial de-identified samples, the architecture comparison used one electrode per architecture, independent fabrication batches and single-aptamer/unpassivated controls are still outstanding, and the lowest calibration point (5 pg mL⁻¹) sits below the calculated LOQ (5.417 pg mL⁻¹). See the manuscript's *Limitations* section for the full statement.
