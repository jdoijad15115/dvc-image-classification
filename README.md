# Hyperparameter Optimization for Image Classification using DVC Experiments

**Author:** Janhavi Doijad (PRN 23070126153), B.Tech AI/ML, SIT Pune

## Task
Image classification of handwritten digits (scikit-learn `digits` dataset: 1,797 grayscale 8×8 images, 10 classes) with an MLP neural network. Hyperparameters are tuned with **DVC Experiments** (`dvc exp run`).

## Project structure
```
dvc-hpo/
├── src/
│   ├── prepare.py      # load images, normalise to [0,1], stratified 75/25 split
│   ├── train.py        # train MLPClassifier using params.yaml
│   └── evaluate.py     # accuracy, macro F1/precision/recall -> metrics.json
├── dvc.yaml            # 3-stage pipeline: prepare -> train -> evaluate
├── dvc.lock            # pinned hashes of the applied (best) run
├── params.yaml         # hyperparameters (now holds the best experiment, exp-13)
├── metrics.json        # metrics of the applied best experiment
├── experiment_results.csv   # all experiments: hyperparameters + metrics
├── requirements.txt
├── .dvc/config, .dvcignore, .gitignore
├── logs/               # raw text output of every command
└── screenshots/        # screenshot of every executed command
```

## Hyperparameters varied
| Param | Values tried |
|---|---|
| `hidden_units` | 16, 32, 64, 128, 256 |
| `activation` | relu, tanh, logistic |
| `learning_rate` | 0.0001, 0.001, 0.002, 0.005, 0.01, 0.05 |
| `alpha` (L2) | 0.0001, 0.001, 0.01, 0.1 |
| `batch_size` | 16, 32, 64, 128 |
| `epochs` | 10, 30, 50, 100, 120, 150 |

Experiments 1–16 vary one hyperparameter at a time from the baseline (64, relu, 0.001, 0.0001, 64, 50); experiments 17–20 are combined configurations.

## Commands executed
```bash
git init && dvc init
dvc repro                                  # baseline -> acc 0.9711
git commit -m "Baseline DVC pipeline"

# 20 experiments (example)
dvc exp run -n exp-13 -S train.hidden_units=64 -S train.activation=relu \
  -S train.learning_rate=0.001 -S train.alpha=0.0001 -S train.batch_size=16 -S train.epochs=50

dvc exp show --sort-by metrics.json:accuracy --sort-order desc
dvc exp show --csv --sort-by metrics.json:accuracy --sort-order desc > experiment_results.csv
dvc exp apply exp-13
dvc metrics show
git commit -m "Apply best experiment exp-13"
dvc params diff HEAD~1
```

## Results
| Rank | Experiment | Accuracy | F1 (macro) | Change vs baseline |
|---|---|---|---|---|
| 1 | **exp-13** | **0.9844** | **0.9844** | batch_size 64 → 16 |
| 1 | exp-04 | 0.9844 | 0.9844 | hidden_units 64 → 256 |
| 1 | exp-08 | 0.9844 | 0.9844 | learning_rate 0.001 → 0.01 |
| 1 | exp-18 | 0.9844 | 0.9842 | 256, lr 0.005, α 0.01, bs 32, 100 ep |
| — | baseline | 0.9711 | 0.9710 | — |
| 20 | exp-07 | 0.8533 | 0.8445 | learning_rate 0.0001 (under-trained) |

**Best experiment: exp-13** (hidden_units=64, relu, lr=0.001, alpha=0.0001, batch_size=16, epochs=50) — accuracy 0.9844, macro-F1 0.9844, +1.33 pts over baseline.

Selection rule: highest accuracy → highest macro-F1 → highest macro-precision → smallest model. Four runs tie on accuracy (443/450 test images correct). exp-18 drops out on F1, exp-08 on precision; exp-04 and exp-13 are identical on all test metrics, and exp-13 wins because it uses a 64-unit hidden layer instead of 256 (~4× fewer parameters) for the same performance.

Note: these top runs differ by at most one test image, so they are statistically equivalent; the tie-break favours the simpler model.

### Observations
- Learning rate matters most: lr=0.0001 under-trains badly in 50 epochs (0.853); 0.001–0.01 works best; 0.05 becomes unstable (0.967).
- Capacity: 16 units under-fits (0.929); returns diminish after 128.
- Activation: relu > tanh > logistic for a fixed budget.
- Smaller batches mean more updates per epoch → better convergence (bs16: 0.984 vs bs128: 0.967).
- Strong L2 (α=0.1) costs some accuracy.

## Reproduce
```bash
pip install -r requirements.txt
dvc repro            # reproduces the applied best experiment
```
On Windows, run the same commands in PowerShell; `python` must be on PATH.

---

# Part 2 — GitHub Actions CI Pipeline for Automated Testing

## Files added
| File | Purpose |
|---|---|
| `.github/workflows/ci.yaml` | CI workflow: triggers on every push (and PR) to `main` or `master`, plus manual `workflow_dispatch` |
| `test.py` | 12 pytest tests covering config, data, model and evaluation stages |
| `requirements.txt` | now includes `pytest` |
| `screenshots/ci_github/` | screenshots of the successful GitHub Actions run |
| `screenshots/ci_local/` | the same steps run locally before pushing |

## CI workflow (`ci.yaml`) — job `test` on `ubuntu-latest`
1. Checkout repository (`actions/checkout@v4`)
2. Set up Python 3.11 with pip cache (`actions/setup-python@v5`)
3. Install dependencies from `requirements.txt`
4. `dvc repro --force` — rebuild the full pipeline from scratch (prepare → train → evaluate); no DVC remote needed since the dataset ships with scikit-learn
5. `dvc metrics show`
6. `python -m pytest test.py -v --junitxml=test-results.xml`
7. Upload `test-results.xml` and `metrics.json` as a build artifact (runs even if tests fail)

## Tests in `test.py`
| Stage | Tests |
|---|---|
| Config | `params.yaml` has valid hyperparameter types/ranges |
| Data | outputs exist; 64 features; train/test split matches `test_size`; pixels normalised to [0,1], no NaNs; all 10 classes in both splits |
| Model | trained model matches `params.yaml`; predictions have the right shape and are valid labels; `predict_proba` rows sum to 1; single-image inference works |
| Evaluation | `metrics.json` has all metrics in [0,1] and agrees with the saved model's real accuracy; **quality gate: accuracy ≥ 0.90**, otherwise the build fails |

The quality gate was checked against a degraded model (lr = 1e-5, 5 epochs → accuracy 0.1356): `test_accuracy_quality_gate` fails and the other 11 pass, so a bad model breaks the build.

## Running locally
```bash
pip install -r requirements.txt
dvc repro
python -m pytest test.py -v      # 12 passed
```

## Pushing to GitHub to trigger CI
```bash
# create an empty repo on github.com first (no README), then:
git remote add origin https://github.com/<username>/dvc-image-classification.git
git push -u origin master
git push origin --all          # optional: also push experiment refs: dvc exp push origin --all
```
Open the repo's **Actions** tab, where the run "CI - Image Classification Tests" starts automatically on the push.



## Actual CI run
Repo: `jdoijad15115/dvc-image-classification` (private), branch `main`. Run #1 (commit `5c21f2c`) finished with **Success in 55s**: all 8 steps green, `dvc repro` reproduced accuracy 0.9844 on the GitHub runner, and pytest reported **12 passed**. Screenshots are in `screenshots/ci_github/`.
