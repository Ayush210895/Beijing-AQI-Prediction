# Beijing AQI Prediction

Machine learning project for estimating Beijing air-quality index (AQI) values and AQI categories from pollutant and weather measurements.

The notebook cleans hourly air-quality records, calculates pollutant-specific AQI components, explores pollution trends, and compares custom implementations of linear regression, logistic regression, and Naive Bayes.

## Project Highlights

- Combines station-level hourly pollution data into one analysis dataset.
- Handles missing values with station/day median imputation and fallback zero fills.
- Converts pollutant concentrations into AQI component indices for PM2.5, PM10, SO2, NO2, CO, and O3.
- Creates AQI categories such as Good, Moderate, Unhealthy, Very Unhealthy, and Hazardous.
- Visualizes AQI distributions, pollutant correlations, and time-based trends.
- Implements machine learning models from scratch for regression and classification.

## Repository Structure

```text
.
+-- IE_7374_Machine_Learning_Project_Group9-1.ipynb  # Main analysis notebook
+-- data/README.md                                   # Dataset download and placement notes
+-- requirements.txt                                 # Python dependencies
+-- README.md                                        # Project documentation
```

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

The analysis compares:

- **Linear Regression** for predicting continuous AQI values.
- **Logistic Regression** for predicting AQI categories.
- **Naive Bayes** for multi-class AQI category prediction.

The implementations are written directly in the notebook, including normalization, training loops, prediction, and evaluation logic.

## Current Results

The saved notebook outputs show the following representative results:

- Linear regression closed-form solution: test RMSE around `49.30`.
- Logistic regression experiments: classification accuracy around `43%` to `47%`, depending on hyperparameters.
- Naive Bayes experiments: classification accuracy around `43%`.

These results are a baseline from the original academic project. A strong next step would be comparing these custom models with scikit-learn baselines and adding cross-validation.

## How to Run

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the dataset

Download `PRSA2017_Data_20130301-20170228.zip` from UCI, unzip it, and place the extracted CSV files in a local folder such as:

```text
data/PRSA_Data_20130301-20170228/
```

### 3. Update the notebook data path

The original notebook was written for Google Colab and uses:

```python
path = "/content/drive/MyDrive/PRSA_Data_20130301-20170228/"
```

For local runs, change it to:

```python
path = "data/PRSA_Data_20130301-20170228/"
```

### 4. Start Jupyter

```bash
jupyter notebook
```

Then open `IE_7374_Machine_Learning_Project_Group9-1.ipynb` and run the cells from top to bottom.

## Original Report

The original final project report is available here:

[IE 7374 - AQI Prediction - Final Project Report](https://github.com/Ayush210895/Beijing-AQI-Prediction/files/9805538/IE.7374.-.AQI.Prediction-.Final.Project.Report-1.pdf)

## Future Improvements

- Add a clean Python script or package version of the notebook workflow.
- Add scikit-learn model baselines for reproducible comparison.
- Save generated charts into a `reports/figures/` folder.
- Add a small sample dataset for smoke testing.
- Add automated notebook execution checks with GitHub Actions.
