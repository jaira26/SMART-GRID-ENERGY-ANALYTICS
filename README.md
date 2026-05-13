# Smart Grid Energy Consumption Analytics with Anomaly Detection

An end-to-end data analytics project that processes smart meter data to detect anomalous energy consumption patterns and forecast next-day electricity demand. The project combines unsupervised anomaly detection with time-series forecasting to help grid operators optimize load distribution and identify potential energy theft or faulty meters.

---

## Overview

The project analyzes 50,000 smart meter readings spanning nearly 4 years from the UCI Individual Household Electric Power Consumption dataset. Two independent analytical pipelines were built: one for anomaly detection using Isolation Forest, and one for demand forecasting using Linear Regression with engineered time-series features.

---

## Key Results

- Achieved 99 percent or higher data quality after automated validation and missing value imputation across 50,000 records
- Isolation Forest flagged 1,253 anomalous cases (5 percent anomaly rate), identifying irregular consumption patterns consistent with energy theft or meter faults
- Linear Regression forecasting model achieved 99.66 percent R-squared on an 80/20 time-ordered train/test split
- Identified peak consumption hours (7 to 9 PM) and 20 percent higher weekend usage, enabling proactive load balancing decisions
- Prophet model also implemented and benchmarked against Linear Regression for comparative forecasting evaluation

---

## Tech Stack

| Component | Technology |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Anomaly Detection | Scikit-learn (Isolation Forest) |
| Demand Forecasting | Scikit-learn (Linear Regression), Prophet |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## Repository Structure

```
smart-grid-energy-analytics/

README.md
data/
    household_sample.csv        # UCI smart meter dataset (sample)
notebooks/
    01_data_exploration.ipynb   # EDA and data quality checks
    02_anomaly_detection.ipynb  # Isolation Forest pipeline
    03_demand_forecasting.ipynb # Linear Regression and Prophet forecasting
outputs/
    detected_anomalies.csv      # Flagged anomalous records
    demand_forecast_results.csv # Model predictions vs actuals
    future_24h_forecast.csv     # Next-day demand forecast
report/
    smart_grid_report.pdf       # Full project write-up
```

---

## Methodology

### Data Preprocessing
- Loaded 50,000+ meter readings with timestamps, voltage, current intensity, and sub-metering measurements
- Applied automated validation checks to flag missing, out-of-range, and inconsistent readings
- Filled missing values using forward fill to preserve temporal continuity
- Achieved 99 percent or higher data quality before modeling

### Anomaly Detection
- Trained an Isolation Forest model on consumption features with a 5 percent contamination parameter
- Model isolates anomalies by randomly partitioning the feature space - irregular readings are isolated faster
- Flagged 1,253 cases classified by anomaly type: high consumption spikes, voltage irregularities, and suspected theft patterns

### Demand Forecasting
- Engineered 17 time-series features including lag values at 1 hour, 24 hours, and 168 hours (1 week)
- Added rolling mean and standard deviation over 24-hour windows
- Applied cyclical encoding (sine and cosine transforms) for hour, day of week, and month
- Used an 80/20 chronological train/test split to preserve temporal order
- Benchmarked Linear Regression against Prophet for 24-hour ahead forecasting

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/jaira26/smart-grid-energy-analytics.git
cd smart-grid-energy-analytics

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn prophet jupyter

# Launch notebooks
jupyter notebook
```

Run notebooks in order: 01 then 02 then 03.

---

## Data Source

Hebrail, G. and Berard, A. (2012). Individual Household Electric Power Consumption. UCI Machine Learning Repository. Retrieved from https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

---

## Author

Jairaghavendra Sridhar
MS Data Analytics Engineering, Northeastern University
https://github.com/jaira26 | https://linkedin.com/in/jairaghavendrasridhar26
