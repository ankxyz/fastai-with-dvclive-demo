# `fast.ai` classification demo with usage of `DVCLive`

**Goal**: use `DVC`/`DVCLive` alongside `fast.ai`.
**Description**: make classification using `fast.ai` and log metrics using `DVCLive`


## Data

**Data source**: https://www.kaggle.com/datasets/tongpython/cat-and-dog/data
**The data in repo**: a little slice of original data split into `train`, `valid` and `test` in the directory `./data/`
**Notes**:
- subset `./data/test` is not using here, but created for future experiments

## Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Run DVC experiment

```bash
dvc exp run
```
