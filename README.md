# Beijing AQI Prediction

Production-ready machine learning pipeline for estimating Beijing air-quality index (AQI) values and AQI categories from pollutant and weather measurements.

The original notebook is still included as an analysis artifact. The main project workflow now lives in a reusable Python package with separate modules for data loading, feature engineering, AQI calculations, model training, and command-line execution.

## Project Highlights

- Loads and combines station-level hourly pollution data into one modeling dataset.
- Handles missing values with station/day median imputation and fallback zero fills.
- Converts pollutant concentrations into AQI component indices for PM2.5, PM10, SO2, NO2, CO, and O3.
- Creates AQI categories such as Good, Moderate, Unhealthy, Very Unhealthy, and Hazardous.
- Trains reproducible baseline models for AQI regression and AQI category classification.
- Saves model artifacts and metrics from a command-line training script.
- Keeps the original notebook available for visual exploration and academic context.

## Repository Structure

```text
.
+-- IE_7374_Machine_Learning_Project_Group9-1.ipynb  # Main analysis notebook
+-- data/README.md                                   # Dataset download and placement notes
+-- pyproject.toml                                   # Package metadata and CLI entry point
+-- requirements.txt                                 # Python dependencies
+-- scripts/download_data.py                         # Official UCI dataset downloader
+-- scripts/train.py                                 # Local training wrapper
+-- src/beijing_aqi/                                 # Production source package
+-- tests/                                           # Lightweight test suite
+-- README.md                                        # Project documentation
```

## Production Package

The `src/beijing_aqi` package separates the notebook into focused modules:

- `aqi.py`: AQI breakpoint formulas, component indices, and category labels.
- `data.py`: raw CSV loading, cleaning, gas unit conversion, rolling-window features, and final AQI feature frame creation.
- `features.py`: model feature/target definitions.
- `models.py`: from-scratch model training and evaluation helpers using NumPy.
- `cli.py`: command-line workflow for training models and writing outputs.

This structure makes the project easier to test, reuse, and extend than a single notebook.

## Dataset

This project uses the Beijing Multi-Site Air Quality dataset from the UCI Machine Learning Repository.

- Source: [Beijing Multi-Site Air Quality - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata)
- File used by the notebook: `PRSA2017_Data_20130301-20170228.zip`
- Time range: March 1, 2013 to February 28, 2017
- Data type: hourly pollutant and meteorological observations across Beijing monitoring stations

The raw CSV files are not committed to this repository because they are external data. See [data/README.md](data/README.md) for setup instructions.

## Methods

### Feature Engineering

The notebook builds the modeling dataset by:

- Reading and concatenating station-level CSV files.
- Creating `Date_time` and `Date` columns from year, month, day, and hour.
- Filling missing pollutant and weather readings by station/day medians.
- Calculating rolling averages and maximums required for AQI-style pollutant indices.
- Converting gas concentrations into ppm or ppb before calculating component indices.
- Assigning final AQI values from the maximum valid pollutant component.

### Models

The production pipeline trains custom models implemented from first principles:

- **Linear Regression** with a closed-form normal-equation solution.
- **Softmax Logistic Regression** with mini-batch gradient descent.
- **Gaussian Naive Bayes** with class priors, means, and variances computed directly from the training data.

The original notebook also includes custom from-scratch implementations used for the classroom project.

## Current Results

After downloading the official UCI dataset, the production pipeline was run on the full dataset of 420,684 model-ready rows.

Full-data training results:

| Model | Metric | Result |
| --- | --- | --- |
| Linear Regression | MAE | `37.77` |
| Linear Regression | RMSE | `49.07` |
| Linear Regression | R2 | `0.53` |
| Softmax Logistic Regression | Accuracy | `0.47` |
| Softmax Logistic Regression | Macro F1 | `0.27` |
| Gaussian Naive Bayes | Accuracy | `0.43` |
| Gaussian Naive Bayes | Macro F1 | `0.39` |

Command used:

```bash
PYTHONPATH=src python3 scripts/train.py \
  --data-dir data/PRSA_Data_20130301-20170228 \
  --output-dir reports
```

The original notebook outputs show the following representative baseline results:

- Linear regression closed-form solution: test RMSE around `49.30`.
- Logistic regression experiments: classification accuracy around `43%` to `47%`, depending on hyperparameters.
- Naive Bayes experiments: classification accuracy around `43%`.

The production results are consistent with the original academic project baseline. A strong next step would be tuning the custom optimization settings and adding cross-validation while keeping the model implementations from scratch.

## How to Run

### 1. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install the package in editable mode:

```bash
pip install -e ".[dev]"
```

### 2. Download the dataset

Download and extract the official UCI dataset:

```bash
python3 scripts/download_data.py
```

This creates:

```text
data/PRSA_Data_20130301-20170228/
```

The dataset is also available on the UCI page:

- [Beijing Multi-Site Air Quality - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata)
- Direct zip: `https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip`

### 3. Update the notebook data path

The original notebook was written for Google Colab and uses:

```python
path = "/content/drive/MyDrive/PRSA_Data_20130301-20170228/"
```

For local runs, change it to:

```python
path = "data/PRSA_Data_20130301-20170228/"
```

### 4. Train models from the command line

```bash
PYTHONPATH=src python3 scripts/train.py \
  --data-dir data/PRSA_Data_20130301-20170228 \
  --output-dir reports
```

The command writes:

- `reports/metrics.json`
- `reports/models/linear_regression.joblib`
- `reports/models/logistic_regression.joblib`
- `reports/models/naive_bayes.joblib`

For a faster smoke run, use:

```bash
PYTHONPATH=src python3 scripts/train.py \
  --data-dir data/PRSA_Data_20130301-20170228 \
  --output-dir reports \
  --sample-size 10000
```

### 5. Run tests

```bash
pytest
```

## Smoke-Test Results

A faster 10,000-row training smoke test produced:

- Linear regression: RMSE `48.32`, R2 `0.56`
- Softmax logistic regression: accuracy `0.47`, macro F1 `0.29`
- Gaussian Naive Bayes: accuracy `0.44`, macro F1 `0.40`

Command used:

```bash
PYTHONPATH=src python3 scripts/train.py \
  --data-dir data/PRSA_Data_20130301-20170228 \
  --output-dir reports \
  --sample-size 10000
```

### 6. Explore the original notebook

```bash
jupyter notebook
```

Then open `IE_7374_Machine_Learning_Project_Group9-1.ipynb` and run the cells from top to bottom.

## Original Report

The original final project report is available here:

[IE 7374 - AQI Prediction - Final Project Report](https://github.com/Ayush210895/Beijing-AQI-Prediction/files/9805538/IE.7374.-.AQI.Prediction-.Final.Project.Report-1.pdf)

## Future Improvements

- Add cross-validation and hyperparameter tuning for the custom models.
- Add model cards or experiment tracking for trained artifacts.
- Save generated charts into a reproducible reporting workflow.
- Add a small sample dataset for smoke testing.
- Add automated notebook execution checks with GitHub Actions.
