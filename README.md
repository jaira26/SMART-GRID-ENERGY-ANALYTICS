Smart Grid Energy Analytics with Real-Time Anomaly Detection

A complete energy consumption analytics project that processes smart meter data to detect anomalies, forecast demand, and provide valuable insights for utilities and consumers.

Project Overview
This project analyzes household electricity consumption data to detect patterns, recogonize the anomalies (energy theft, faulty meters), and forecast future demand. The system is configured for smart grid applications where real-time monitoring and predictive analytics are crucial for grid stability and operational efficiency.

Key Features
Data Exploration & Visualization: Comprehensive analysis of consumption patterns across time 
Anomaly Detection: Machine learning-based identification of unusual consumption patterns using Isolation Forest
Demand Forecasting: Time-series prediction of future energy demand using regression models
Real-Time Analytics: Framework designed for edge computing deployment

Technologies Used
Python 3.x
pandas - Data manipulation and analysis
NumPy - Numerical computing
scikit-learn - Machine learning (Isolation Forest, Linear Regression)
Matplotlib & Seaborn - Data visualization
Power BI - Dashboard creation (In Progress)

Dataset
The project utilizes household electricity consumption data with minute-level granularity including:
Global active/reactive power
Voltage and current intensity
Sub-metering for kitchen, laundry, and HVAC appliances

```smart-grid-energy-analytics/
├── data/
│   └── household_sample.csv
├── 01_data_exploration.py
├── 02_anomaly_detection.py
├── 03_demand_forecasting.py
└── README.md
```

Installation & Setup
Clone the repository:
git clone https://github.com/jaira26/smart-grid-energy-analytics.git
cd smart-grid-energy-analytics

Install required packages:
pip install pandas numpy matplotlib seaborn scikit-learn

Usage
1. Data Exploration
bashpython 01_data_exploration.py

Outputs:
Statistical summary of consumption patterns
Temporal analysis (hourly, daily, weekly patterns)
Sub-metering breakdown
4 visualization plots

2. Anomaly Detection
bashpython 02_anomaly_detection.py

Outputs:
Detected anomalies with severity scores
Classification of anomaly types (high consumption, voltage issues, potential theft)
5 visualization plots
CSV files: detected_anomalies.csv, energy_data_with_anomalies.csv

3. Demand Forecasting
bashpython 03_demand_forecasting.py

Outputs:
24-hour demand forecast
Model performance metrics (R², MAE, RMSE)
Feature importance analysis
7 visualization plots
CSV files: demand_forecast_results.csv, future_24h_forecast.csv

Key Results
Anomaly Detection Performance
Detection rate: ~5% of data flagged as anomalous
Identifies high consumption spikes, voltage anomalies, and potential energy theft
Temporal patterns reveal peak anomaly hours

Demand Forecasting Performance
Model: Linear Regression with time-series features
Prediction accuracy: 90%+ (varies by dataset)
Key predictors: Lag features, hour of day, rolling averages

Business Applications
Utilities: Grid load balancing, fault detection, theft prevention
Consumers: Consumption monitoring, cost optimization
Smart Cities: Real-time infrastructure monitoring
Energy Management: Demand response programs, dynamic pricing

Future Enhancements
Deploy LSTM models for improved long-term forecasting
Implement real-time streaming analytics with Apache Kafka
Create interactive Power BI dashboard
Add renewable energy integration analysis
Deploy model as REST API for edge devices

Technical Highlights
Time-series Feature Engineering: Lag features, rolling statistics, cyclical encoding
Unsupervised ML: Isolation Forest for anomaly detection without labeled data
Scalable Architecture: Modular design suitable for production deployment
Edge Computing Ready: Lightweight models optimized for distributed processing

Author
Jairaghavendra Sridhar

Data Analytics Engineering
LinkedIn (www.linkedin.com/in/jairaghavendrasridhar26)

License
This project is available for educational and portfolio purposes.

Acknowledgments
Dataset: Individual household electric power consumption from UCI Machine Learning Repository


