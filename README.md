# DA3408 — AI Operations: Module 1 Assignment

Experiment management and reproducibility with **MLflow** and **DVC**.

| Question | Topic | Where |
| --- | --- | --- |
| Q1 | Technical-debt diagnosis (written) | [`q1/aiops_q1.pdf`](q1/aiops_q1.pdf) |
| Q2 | MLflow experiment comparison | [`q2/Q2_Mlflow.ipynb`](q2/Q2_Mlflow.ipynb) |
| Q3 | DVC data versioning & rollback | repo root (`data.dvc`, `file_list.csv.dvc`, tags `v1`/`v2`) |
| Q4 | End-to-end reproducibility drill (capstone) | [`q4/question_4.ipynb`](q4/question_4.ipynb) — Partner A's repo: <https://github.com/samrudh123/q4> |

Summary write-up: [`assignment1_report.pdf`](assignment1_report.pdf).
Screenshots referenced below live in [`screenshots/`](screenshots/).

---

## Environment

Two environments are used, because Q4 has to be reproduced from a pinned spec file:

```bash
# Q1–Q3: the general virtualenv
virtualenv ~/aiops
source ~/aiops/bin/activate
pip install "dvc[all]" mlflow scikit-learn pandas ipykernel

# Q4: the exact environment Partner A pinned (see q4/environment.yml)
conda env create -f q4/environment.yml
conda activate aiops-q4
```

> `dvc` is only on `PATH` inside one of these environments. Activating first is not optional —
> in the Q4 recording, `dvc pull` fails with *"Command 'dvc' not found"* until `conda activate
> aiops-q4` is run.

---

## Q1 — Technical debt diagnosis

Written answer only, in [`q1/aiops_q1.pdf`](q1/aiops_q1.pdf).
It maps (a) the rounding change that broke an unrelated feature to **boundary erosion /
entanglement (CACE)**, (b) the unknown dashboard team to **undeclared consumers (visibility
debt)**, and (c) the 14 undocumented shell scripts to **pipeline jungle**, then proposes
rewriting (c) as a `dvc.yaml` DAG with `dvc repro`, `params.yaml`, and MLflow logging per run.

Nothing to execute.

---

## Q2 — MLflow experiment comparison

Six MLP runs on MNIST over a 3 × 2 grid: `hidden_layer_sizes` ∈ {(8,), (16,), (32,)} ×
`learning_rate_init` ∈ {0.001, 0.01}, 30 epochs each, tracked on a local MLflow server.

### 1. Start the tracking server

Run this **from `q2/`** in a separate terminal and leave it running:

```bash
source ~/aiops/bin/activate
cd q2
mlflow server \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 4000 \
  --allowed-hosts "*" \
  --cors-allowed-origins "http://localhost:4000, http://127.0.0.1:4000"
```

Port **4000** matters: the notebook calls `mlflow.set_tracking_uri("http://localhost:4000")`.
The UI is then at <http://localhost:4000>.

### 2. Run the notebook

Open [`q2/Q2_Mlflow.ipynb`](q2/Q2_Mlflow.ipynb) and run all cells top to bottom. Working
directory must be `q2/`, so that the dataset resolves — the notebook looks for
`data/mnist_784.npz` (committed at `q2/data/mnist_784.npz`) and only falls back to
`fetch_openml("mnist_784")` if it is missing.

What each step does:

| Step | Cell does |
| --- | --- |
| 0 | installs deps, points MLflow at `localhost:4000`, sets experiment `mnist-mlp` |
| 1 | loads MNIST, scales to `[0,1]`, subsamples 12 000 rows, 80/20 stratified split → 9 600 train / 2 400 val |
| 2 | the un-instrumented starter `train_and_evaluate()` — `MLPClassifier` with `max_iter=1, warm_start=True` so each `.fit()` is exactly one epoch |
| 3 | `train_and_log()` — the same loop wrapped in `mlflow.start_run()` |
| 4 | the sweep: the 3 × 2 grid, six runs named `mlp-<width>-lr<rate>` |
| 5 | `mlflow.search_runs(...)` → the run-comparison table, sorted by `final_val_accuracy` |
| 6 | rebuilds per-epoch curves with `client.get_metric_history()` to show the overfitting evidence |
| 7 | pivots final accuracy on architecture × learning rate to compare the two effects |

### 3. The logging code that was added (Q2.3)

```python
with mlflow.start_run(run_name=run_name):
    mlflow.log_param("hidden_layer_sizes", str(hidden_layer_sizes))
    mlflow.log_param("learning_rate_init", learning_rate_init)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)

    # per-epoch curves — `step=` is what makes MLflow draw a line chart, not a point
    for h in history:
        mlflow.log_metric("train_loss",     h["train_loss"],     step=h["epoch"])
        mlflow.log_metric("train_accuracy", h["train_accuracy"], step=h["epoch"])
        mlflow.log_metric("val_accuracy",   h["val_accuracy"],   step=h["epoch"])
        mlflow.log_metric("val_f1_macro",   h["val_f1_macro"],   step=h["epoch"])

    # summary metrics — these are the columns the comparison table ranks on
    mlflow.log_metric("final_train_loss",   final["train_loss"])
    mlflow.log_metric("final_val_accuracy", final["val_accuracy"])
    mlflow.log_metric("final_val_f1_macro", final["val_f1_macro"])
    mlflow.log_metric("best_val_accuracy",  best["val_accuracy"])
    mlflow.log_metric("best_epoch",         best["epoch"])
    mlflow.log_metric("overfit_gap", final["train_accuracy"] - final["val_accuracy"])
    mlflow.sklearn.log_model(model, name="model")
```

### 4. Reproducing the comparison screenshot

In the MLflow UI: **Experiments → `mnist-mlp` → select all six runs → Compare**, then toggle
*Show diff only* on both the Parameters and Metrics panels. That is
[`screenshots/q2_results.png`](screenshots/q2_results.png).

Result: best run is `mlp-32-lr0.001` at **0.9313** val accuracy. The written 150–250 word
analysis is the final *Analysis* markdown cell of the notebook.

---

## Q3 — DVC data versioning & rollback

Run from the **repository root** (that is where `data.dvc`, `file_list.csv.dvc` and the `v1`/`v2`
tags live). Everything below is what was actually executed; `data/` and `file_list.csv` are
DVC-tracked, so Git only stores the small `.dvc` pointers.

```bash
source ~/aiops/bin/activate
```

### Q3.1 — Init, build the CSV, track it, push v1 to an SSH remote

```bash
dvc init
git commit -m "Initialise DVC"

# the class dataset from Lecture 2
dvc get https://github.com/iterative/dataset-registry tutorials/versioning/data.zip
unzip data.zip && rm -f data.zip
tree -d data                                    # train/{cats,dogs}, validation/{cats,dogs}
find data -type f -name "*.jpg" | wc -l         # 1800

# one header line + one row per image
echo -e "filepath\n$(find data -type f -name "*.jpg")" > file_list.csv
wc -l file_list.csv                             # 1801  = 1800 rows + header

dvc add data file_list.csv
cat data.dvc                                    # md5 pointer Git will store
cat .gitignore                                  # DVC added /data and /file_list.csv
```

Configure the **SSH remote** (key auth first, so `dvc push` is not prompted for a password):

```bash
ssh-keygen -t ed25519 -C "samrudhraaj@gmail.com"          # once, if you have no key
ssh-copy-id -i ~/.ssh/id_ed25519.pub samrudh@10.17.64.87
ssh samrudh@10.17.64.87 "mkdir -p /research/samrudh/dvcstore/da3408 && ls -ld /research/samrudh/dvcstore/da3408"

# register it as the default remote  (-d / --default)
dvc remote add -d ssh-remote ssh://samrudh@10.17.64.87/research/samrudh/dvcstore/da3408
dvc remote modify ssh-remote user samrudh
dvc remote modify --local ssh-remote keyfile ~/.ssh/id_ed25519
# if key auth is unavailable, prompt for the password instead:
# dvc remote modify --local ssh-remote ask_password true

dvc remote list
cat .dvc/config
git add .dvc/config && git commit -m "Add SSH DVC remote"
```

> `--local` writes to `.dvc/config.local`, which is git-ignored — that is where the key path or
> password prompt belongs, so no credentials land in the shared `.dvc/config`.
> Substitute your own host/path; `10.17.64.87:/research/samrudh` is the lab machine used here.

Commit and push **v1**:

```bash
git add .gitignore data.dvc file_list.csv.dvc
git commit -m "data v1 added with 1800 files"
git tag -a v1 -m "v1 with 1800 images"

dvc status -c          # cloud status: everything is "new", nothing on the remote yet
dvc push -j 8          # uploads the cache to 10.17.64.87
dvc status -c          # -> "Cache and remote 'ssh-remote' are in sync."

# proof the bytes landed on the remote machine
ssh samrudh@10.17.64.87 "find /research/samrudh/dvcstore/da3408 -type f | wc -l; du -sh /research/samrudh/dvcstore/da3408"
```

### Q3.2 — Simulate the data update → v2

```bash
dvc get https://github.com/iterative/dataset-registry tutorials/versioning/new-labels.zip
unzip -o new-labels.zip && rm -f new-labels.zip
find data -type f -name "*.jpg" | wc -l         # 2800  (1000 new training images)

dvc status                                      # data/ is modified, not yet re-added
dvc diff

# regenerate the CSV from the updated dataset
echo -e "filepath\n$(find data -type f -name "*.jpg")" > file_list.csv
wc -l file_list.csv                             # 2801  = 2800 rows + header

dvc add data file_list.csv
git diff data.dvc file_list.csv.dvc             # the md5s changed -> screenshots/q3_git_diff.png

git commit -am "data v2 with 2800 files"
git tag -a v2 -m "v2 with 2800 images"

dvc push -j 8                                   # push the v2 cache objects
```

The pointer diff — `1800 → 2800` files, `54993 → 82995` bytes of CSV — is
[`screenshots/q3_git_diff.png`](screenshots/q3_git_diff.png).

### Q3.3 — Roll back to v1 with `git checkout` + `dvc checkout`

```bash
# BEFORE (v2)
wc -l file_list.csv                             # 2801
find data -type f -name "*.jpg" | wc -l         # 2800

git checkout v1                                 # moves the .dvc POINTERS back; files untouched
dvc status                                      # data.dvc / file_list.csv.dvc: modified
dvc checkout                                    # restores the actual bytes from the cache
dvc status                                      # -> "Data and pipelines are up to date."

# AFTER (v1)  <-- proof
wc -l file_list.csv                             # 1801
find data -type f -name "*.jpg" | wc -l         # 1800
```

Terminal proof: [`screenshots/q3_checkout_diffs.png`](screenshots/q3_checkout_diffs.png).
The two commands are both needed — `git checkout` alone only rewinds the hash in the `.dvc`
file, and `dvc checkout` is what swaps the workspace data to match it.

Return to the latest version:

```bash
git checkout main
dvc checkout
git push origin main --tags
```

---

## Q4 — End-to-end reproducibility drill (capstone)

Two partners, one protocol. **Partner A** trained the baseline and pushed the repository;
**Partner B** reproduced it from the repo alone, with no other communication about the
environment or the data. Partner A's work arrives here as the `q4/` subtree
(`git subtree pull --prefix=q4 team main`); Partner B's side is the recording,
`../q4_video.mp4`.

**Partner A's repository: <https://github.com/samrudh123/q4>**

The subtree merge grafts Partner A's tree into `q4/` here, so their individual Q4 commits are
browsable in that repository rather than in this one's history. It is wired up as the `team`
remote:

```bash
git remote add team https://github.com/samrudh123/q4.git
git fetch team
git log team/main --oneline                              # Partner A's commit history
git show team/main:question_4.ipynb                      # a file at Partner A's commit

# how q4/ got here, and how it is re-synced
git subtree add  --prefix=q4 team main
git subtree pull --prefix=q4 team main -m "sync partner work"
```

Everything for Q4 runs **inside `q4/`**, which is its own DVC project with an **S3** remote
(`s3://aiops-dvcstore/q4`, `ap-south-1`) — separate from the SSH remote used by Q3.

### Partner A — baseline (Q4.1)

Trains an `MLPClassifier((128, 64))` on the full 70 000-row MNIST with `seed=42`, and in the
same run logs the params, the metrics, the per-epoch `train_loss` / `val_accuracy` curves, and
the provenance tags that make the run re-findable:

```python
mlflow.set_tag("git_commit", git_commit)          # from `git rev-parse HEAD`
mlflow.set_tag("git_dirty", git_dirty)            # true = the commit alone won't reproduce this
mlflow.set_tag("dataset_md5", data_md5)           # md5 of the bytes actually trained on
mlflow.set_tag("dataset_dvc_md5", dvc_md5)        # the md5 DVC recorded in data.dvc
mlflow.set_tag("sklearn_version", sklearn.__version__)
mlflow.set_tag("python_version", platform.python_version())
```

It then registers the model as `mnist-mlp-q4` and transitions the new version to **Staging**
(falling back to the `staging` *alias* if the server is MLflow 3, where stages are deprecated),
and writes [`q4/sujal_run.json`](q4/sujal_run.json) — the baseline record, with `run_id`,
`git_commit`, `dataset_md5`, library versions, params, metrics and a stated tolerance of
**±0.005**. The dataset is versioned with `dvc add data` and `q4/data.dvc` is committed in the
**same** Git commit as the code, then `dvc push`ed to S3.

Baseline: `run_id b0461ee0…`, **accuracy 0.9751**, **f1_macro 0.9750**.

### Partner B — reproduction (Q4.2–Q4.4)

This is the sequence in `q4_video.mp4`, using only the five allowed operations:

```bash
git clone https://github.com/samrudh123/DA3408-Assignment-1.git
cd DA3408-Assignment-1/q4
git checkout <commit>                       # the commit that carries the code + data.dvc

conda env create -f environment.yml         # python 3.13, sklearn 1.9.0, mlflow 3.15.1, dvc[all]
conda activate aiops-q4                     # <- dvc is NOT on PATH before this

aws configure                               # credentials for the s3://aiops-dvcstore remote
dvc pull                                    # "A  data/  1 file added"  -> restores mnist_784.npz

mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

The `sqlite` backend is required — the Model Registry does not work with a plain file store.
Port **5000** matches `mlflow.set_tracking_uri("http://localhost:5000")` in the notebook; if you
serve on another port, change the notebook to match.

Then open [`q4/question_4.ipynb`](q4/question_4.ipynb), pick the **`aiops-q4`** kernel, and
**Run All**. The notebook branches on whether `sujal_run.json` already exists:

- absent → this is the **baseline**; the run is named `baseline` and the file is written.
- present → this is a **reproduction**; the run is named `reproduction`, and Step 6 compares
  the new metrics against the baseline within the stated tolerance.

Step 6 writes the comparison as the run description and as tags
(`repro_verdict`, `repro_tolerance`, `repro_delta_accuracy`, `repro_delta_f1_macro`) — that is
the Q4.4 note. Result:

| metric | baseline | reproduction | delta | within ±0.005 |
| --- | --- | --- | --- | --- |
| accuracy | 0.9751 | 0.9751 | +0.0000 | yes |
| f1_macro | 0.9750 | 0.9750 | +0.0000 | yes |

**Verdict: REPRODUCED.** `dataset_md5` and `seed` match, `sklearn_version` matches at 1.9.0 —
the two differences are `git_commit` (Partner B is on a later commit that also contains
Partner B's own files) and `python_version` (3.13.14 vs 3.13.15, a patch release), neither of
which moved the metrics. The reproduction run is registered as `mnist-mlp-q4` **v3**.

MLflow run view: [`screenshots/q4_partnerb_results.jpeg`](screenshots/q4_partnerb_results.jpeg).

---

## Repository layout

```
.
├── README.md
├── assignment1_report.pdf       # the 1-page write-up
├── q1/aiops_q1.pdf              # Q1 written answer (+ .tex source)
├── q2/
│   ├── Q2_Mlflow.ipynb          # Q2 sweep + analysis
│   ├── data/mnist_784.npz
│   └── mlruns/, models/         # MLflow artifacts for the six runs
├── q3/q3_dvc_commands.sh        # earlier draft script; the procedure above is the one that ran
├── q4/                          # subtree of github.com/samrudh123/q4 — own DVC project (S3)
│   ├── question_4.ipynb
│   ├── environment.yml          # the pinned spec Partner B recreates
│   ├── data.dvc                 # committed with the code, in the same commit
│   └── sujal_run.json           # baseline record + tolerance
├── data.dvc                     # Q3: 2800-image dataset pointer  (v2)
├── file_list.csv.dvc            # Q3: CSV pointer                 (v2)
└── screenshots/
    ├── q2_results.png           # MLflow six-run comparison table
    ├── q3_git_diff.png          # v1 -> v2 pointer diff
    ├── q3_checkout_diffs.png    # rollback proof, 2801 -> 1801 lines
    └── q4_partnerb_results.jpeg # reproduction run in MLflow
```

Git tags `v1` (1800 images) and `v2` (2800 images) mark the two Q3 dataset versions.
