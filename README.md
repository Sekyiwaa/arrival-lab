# Arrival Lab

A small bus arrival research prototype by Akua Sekyiwaa. Replay public GPS journeys and compare predicted remaining travel time with the recorded outcome.

## Current status

The visual demo and baseline benchmark work. TabPFN-3.5 Plus integration is implemented, but has not been evaluated because this build has no authenticated Prior Labs account. No TabPFN scores or predictions are fabricated. This version is not yet a completed TabPFN hackathon entry.

## Run locally

Use Python 3.10 or later. Open a terminal in this folder:

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python benchmark.py
python -m http.server 8000 --directory dist
```

Open https://sekyiwaa.github.io/arrival-lab/. Select a journey, choose a model, and move the replay slider or press Play.

## Evaluate TabPFN-3.5

Authenticate using the official tabpfn-client account flow on your own computer. Do not put tokens into the repository or browser code. Then run:

```sh
python -c "from tabpfn_client import interactive_login; interactive_login()"
python benchmark.py --tabpfn
```

This selects `TabPFNRegressor.create_default_for_version("v3.5")` explicitly. A successful run writes genuine predictions and metrics into `dist/results.json`. Refresh the local demo to select TabPFN-3.5 Plus. The client may prompt for authentication; account usage limits apply. See https://docs.priorlabs.ai/getting-started/quickstart and https://docs.priorlabs.ai/models/selecting-model-version.

## Prediction task

For an observation during a bus journey, predict minutes until the final recorded GPS point. The destination coordinates are assumed to be known in advance. The task is arrival at a recorded endpoint, which may not be the route terminal or a school.

Inputs: current latitude/longitude, destination coordinates, direct distance, elapsed minutes, recent speed estimated only from past GPS points, local hour and weekday. The target is future remaining time. Full-trip average speed, total distance, total duration and subjective trip ratings are excluded.

Filtering keeps bus-labeled tracks with at least 15 distinct timestamps and recorded durations between 3 and 180 minutes. Observations begin after six GPS points and exclude the last 30 seconds. Sampling limits dense trips to approximately 40 observations. There are 41 eligible journeys after filtering, 30 training journeys and 11 test journeys, using a fixed seed of 42. No journey appears in both partitions.

## Evaluation

Compare an average-speed baseline, random forest and optionally TabPFN on the same test observations. Report observation-weighted mean absolute error, trip-balanced mean absolute error, and percentage within two minutes. Lower error is better. Whole journeys are the independent evaluation units.

The initial baseline run reports mean errors of 8.17 minutes for the average-speed baseline and 6.56 minutes for random forest. The average-speed baseline has a higher share within two minutes, so random forest is not uniformly better. These are exploratory single-split results, not a general performance claim. API timings include network overhead.

Before a stronger submission, add trip-level repeated evaluation or bootstrapped confidence intervals, a tuned tree baseline, and more journeys. Keep test data out of model selection.

## Data and limitations

Source: Cruz, M., Macedo, H., Barreto, R., & Guimares, A. GPS Trajectories. UCI Machine Learning Repository. https://doi.org/10.24432/C54S5Z

Source page: https://archive.ics.uci.edu/dataset/354/gps+trajectories

Original CSV files and source metadata are included. Data contains Brazilian journeys and timestamps documented as GMT-3. It contains no Ghana pilot data. The repository does not establish school bus safety, boarding confirmation, live location, or accuracy for Ghanaian routes. Replays expose the complete historical route for visualization, but model inputs never contain future GPS points apart from the explicit known-destination assumption.

The dataset license should be checked on the source page before public redistribution or hackathon submission. This package includes original attribution; code is provided under the MIT license.

## Files

- `benchmark.py`: preprocessing, trip split, benchmark and prediction export.
- `dist/index.html`: responsive visual replay and results dashboard.
- `dist/results.json`: actual baseline output, replaced after a successful TabPFN run.
- `data/`: original public CSVs and UCI metadata.
- `requirements.txt`: dependencies.
