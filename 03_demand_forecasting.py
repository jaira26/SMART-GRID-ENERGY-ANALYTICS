"""
Smart Grid Energy Analytics - Demand Forecasting
Author: Jairaghavendra Sridhar
Purpose: Predict future energy demand using time-series analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SMART GRID DEMAND FORECASTING SYSTEM")
print("="*70)


# STEP 1: LOAD AND PREPARE DATA

print("\nSTEP 1: Loading Data...")

try:
    df = pd.read_csv('data/household_sample.csv')
    print(f"Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print("Error: household_sample.csv not found in data/ folder")
    exit()

# Create timestamp
df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                  format='%d/%m/%Y %H:%M:%S')

# Sort by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Handle missing values
df = df.fillna(method='ffill').fillna(method='bfill')

print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")


# STEP 2: FEATURE ENGINEERING FOR FORECASTING

print("\nSTEP 2: Engineering Time Features...")

# Extract time components
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['day_of_year'] = df['timestamp'].dt.dayofyear

# Create cyclical features (important for time-series)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Lag features (previous values)
df['power_lag_1h'] = df['Global_active_power'].shift(60)  # 1 hour ago (60 minutes)
df['power_lag_24h'] = df['Global_active_power'].shift(1440)  # 24 hours ago
df['power_lag_168h'] = df['Global_active_power'].shift(10080)  # 1 week ago

# Rolling statistics (moving averages)
df['power_rolling_mean_24h'] = df['Global_active_power'].rolling(window=1440, min_periods=1).mean()
df['power_rolling_std_24h'] = df['Global_active_power'].rolling(window=1440, min_periods=1).std()

# Drop rows with NaN from lag features
df_model = df.dropna().reset_index(drop=True)

print(f"Created time-series features")
print(f"Model dataset: {len(df_model):,} records (after removing NaN)")


# STEP 3: PREPARE TRAIN/TEST SPLIT

print("\nSTEP 3: Splitting Data...")

# Use 80% for training, 20% for testing (time-ordered split)
split_idx = int(len(df_model) * 0.8)

train = df_model.iloc[:split_idx].copy()
test = df_model.iloc[split_idx:].copy()

print(f"Training set: {len(train):,} records")
print(f"   From: {train['timestamp'].min()}")
print(f"   To: {train['timestamp'].max()}")
print(f"\nTest set: {len(test):,} records")
print(f"   From: {test['timestamp'].min()}")
print(f"   To: {test['timestamp'].max()}")

# Define features for modeling
feature_cols = [
    'hour', 'day_of_week', 'month', 'is_weekend',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
    'power_lag_1h', 'power_lag_24h', 'power_lag_168h',
    'power_rolling_mean_24h', 'power_rolling_std_24h',
    'Voltage', 'Global_intensity'
]

X_train = train[feature_cols]
y_train = train['Global_active_power']

X_test = test[feature_cols]
y_test = test['Global_active_power']

print(f"\nFeatures used: {len(feature_cols)}")


# STEP 4: TRAIN FORECASTING MODEL

print("\nSTEP 4: Training Forecasting Model...")

# Linear Regression model
model = LinearRegression()

print("Training model...")
model.fit(X_train, y_train)

print("Model trained successfully!")

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# STEP 5: EVALUATE MODEL PERFORMANCE

print("\nSTEP 5: Evaluating Model Performance...")

# Training metrics
train_mae = mean_absolute_error(y_train, train_predictions)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
train_r2 = r2_score(y_train, train_predictions)

# Testing metrics
test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_r2 = r2_score(y_test, test_predictions)

print("\n" + "="*70)
print("MODEL PERFORMANCE METRICS")
print("="*70)

print("\nTRAINING SET:")
print(f"   MAE (Mean Absolute Error): {train_mae:.4f} kW")
print(f"   RMSE (Root Mean Squared Error): {train_rmse:.4f} kW")
print(f"   R² Score: {train_r2:.4f}")

print("\nTEST SET:")
print(f"   MAE (Mean Absolute Error): {test_mae:.4f} kW")
print(f"   RMSE (Root Mean Squared Error): {test_rmse:.4f} kW")
print(f"   R² Score: {test_r2:.4f}")

# Calculate percentage errors
mean_actual = y_test.mean()
mape = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100

print(f"\nADDITIONAL METRICS:")
print(f"   Mean Actual Power: {mean_actual:.4f} kW")
print(f"   MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
print(f"   Prediction Accuracy: {100 - mape:.2f}%")


# STEP 6: FEATURE IMPORTANCE

print("\nSTEP 6: Feature Importance Analysis...")

# Get feature coefficients
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
})

feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print("\nTOP 10 MOST IMPORTANT FEATURES:")
print(feature_importance.head(10).to_string(index=False))


# STEP 7: VISUALIZATIONS

print("\nSTEP 7: Creating Visualizations...")

plt.style.use('seaborn-v0_8-darkgrid')

# Figure 1: Actual vs Predicted (Training)
print("   Creating Figure 1: Training predictions...")
fig, ax = plt.subplots(figsize=(15, 6))

# Sample every 100th point for clarity
sample_train = train.iloc[::100]
sample_train_pred = train_predictions[::100]

ax.plot(sample_train['timestamp'], sample_train['Global_active_power'], 
        label='Actual', linewidth=1.5, alpha=0.7, color='blue')
ax.plot(sample_train['timestamp'], sample_train_pred, 
        label='Predicted', linewidth=1.5, alpha=0.7, color='red', linestyle='--')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Power (kW)', fontsize=12)
ax.set_title(f'Training Set: Actual vs Predicted (R² = {train_r2:.4f})', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Figure 2: Actual vs Predicted (Testing)
print("   Creating Figure 2: Test predictions...")
fig, ax = plt.subplots(figsize=(15, 6))

# Sample every 50th point
sample_test = test.iloc[::50]
sample_test_pred = test_predictions[::50]

ax.plot(sample_test['timestamp'], sample_test['Global_active_power'], 
        label='Actual', linewidth=2, alpha=0.8, color='green')
ax.plot(sample_test['timestamp'], sample_test_pred, 
        label='Predicted', linewidth=2, alpha=0.8, color='orange', linestyle='--')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Power (kW)', fontsize=12)
ax.set_title(f'Test Set: Actual vs Predicted (R² = {test_r2:.4f})', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Figure 3: Prediction Errors
print("   Creating Figure 3: Prediction error distribution...")
test_errors = y_test - test_predictions

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(test_errors, bins=50, color='coral', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax.set_xlabel('Prediction Error (kW)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Prediction Errors', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 4: Scatter Plot - Actual vs Predicted
print("   Creating Figure 4: Actual vs predicted scatter...")
fig, ax = plt.subplots(figsize=(10, 10))

# Sample for visualization
sample_indices = np.random.choice(len(test), size=min(5000, len(test)), replace=False)
y_test_sample = y_test.iloc[sample_indices]
test_pred_sample = test_predictions[sample_indices]

ax.scatter(y_test_sample, test_pred_sample, alpha=0.5, s=20, color='steelblue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Power (kW)', fontsize=12)
ax.set_ylabel('Predicted Power (kW)', fontsize=12)
ax.set_title(f'Actual vs Predicted Power (R² = {test_r2:.4f})', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 5: Feature Importance
print("   Creating Figure 5: Feature importance...")
fig, ax = plt.subplots(figsize=(12, 8))
top_features = feature_importance.head(15)
ax.barh(top_features['Feature'], top_features['Abs_Coefficient'], 
        color='teal', alpha=0.7, edgecolor='black')
ax.set_xlabel('Absolute Coefficient Value', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('Top 15 Most Important Features', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# Figure 6: Hourly Forecast Comparison
print("   Creating Figure 6: Hourly forecast comparison...")
hourly_actual = test.groupby('hour')['Global_active_power'].mean()
test['predictions'] = test_predictions
hourly_predicted = test.groupby('hour')['predictions'].mean()

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(24)
width = 0.35

ax.bar(x - width/2, hourly_actual.values, width, label='Actual', 
       color='skyblue', alpha=0.8, edgecolor='black')
ax.bar(x + width/2, hourly_predicted.values, width, label='Predicted', 
       color='salmon', alpha=0.8, edgecolor='black')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Average Power (kW)', fontsize=12)
ax.set_title('Average Hourly Consumption: Actual vs Predicted', 
             fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


# STEP 8: SAVE RESULTS

print("\nSTEP 8: Saving Results...")

# Save predictions
forecast_results = pd.DataFrame({
    'timestamp': test['timestamp'],
    'actual_power': y_test.values,
    'predicted_power': test_predictions,
    'error': y_test.values - test_predictions,
    'hour': test['hour'],
    'day_of_week': test['day_of_week'],
    'is_weekend': test['is_weekend']
})

forecast_results.to_csv('data/demand_forecast_results.csv', index=False)
print(f"Saved: data/demand_forecast_results.csv ({len(forecast_results):,} records)")

# Save feature importance
feature_importance.to_csv('data/feature_importance.csv', index=False)
print(f"Saved: data/feature_importance.csv")

# STEP 9: GENERATE FUTURE FORECAST

print("\nSTEP 9: Generating Future Forecast...")

# Generate next 24 hours forecast
last_timestamp = df_model['timestamp'].max()
future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=1), 
                                   periods=1440, freq='1min')  # Next 24 hours

print(f"Forecasting period: {future_timestamps[0]} to {future_timestamps[-1]}")

# Create future dataframe with same features
future_df = pd.DataFrame({'timestamp': future_timestamps})
future_df['year'] = future_df['timestamp'].dt.year
future_df['month'] = future_df['timestamp'].dt.month
future_df['day'] = future_df['timestamp'].dt.day
future_df['hour'] = future_df['timestamp'].dt.hour
future_df['day_of_week'] = future_df['timestamp'].dt.dayofweek
future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)

# Cyclical features
future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour'] / 24)
future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour'] / 24)
future_df['day_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
future_df['day_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)

# Use last known values for lag features (simplified approach)
last_power = df_model['Global_active_power'].iloc[-1]
last_voltage = df_model['Voltage'].iloc[-1]
last_intensity = df_model['Global_intensity'].iloc[-1]

future_df['power_lag_1h'] = last_power
future_df['power_lag_24h'] = last_power
future_df['power_lag_168h'] = last_power
future_df['power_rolling_mean_24h'] = df_model['power_rolling_mean_24h'].iloc[-1]
future_df['power_rolling_std_24h'] = df_model['power_rolling_std_24h'].iloc[-1]
future_df['Voltage'] = last_voltage
future_df['Global_intensity'] = last_intensity

# Make future predictions
X_future = future_df[feature_cols]
future_predictions = model.predict(X_future)

# Save future forecast
future_forecast = pd.DataFrame({
    'timestamp': future_timestamps,
    'predicted_power': future_predictions,
    'hour': future_df['hour']
})

future_forecast.to_csv('data/future_24h_forecast.csv', index=False)
print(f"Saved: data/future_24h_forecast.csv (next 24 hours)")

# Visualize future forecast
print("   Creating future forecast visualization...")
fig, ax = plt.subplots(figsize=(15, 6))

# Plot last 3 days of actual data
recent_data = df_model.tail(4320)  # Last 3 days
ax.plot(recent_data['timestamp'], recent_data['Global_active_power'], 
        label='Historical Data', linewidth=1.5, color='blue', alpha=0.7)

# Plot future forecast
ax.plot(future_timestamps, future_predictions, 
        label='24h Forecast', linewidth=2, color='red', linestyle='--', alpha=0.8)

ax.axvline(last_timestamp, color='green', linestyle=':', linewidth=2, 
           label='Forecast Start')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Power (kW)', fontsize=12)
ax.set_title('Energy Demand Forecast: Next 24 Hours', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# STEP 10: KEY INSIGHTS & RECOMMENDATIONS

print("\n" + "="*70)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*70)

print(f"\n1. MODEL PERFORMANCE:")
print(f"   Test R² Score: {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
print(f"   Prediction Accuracy: {100 - mape:.2f}%")
print(f"   Average Error: {test_mae:.4f} kW")

print(f"\n2. KEY FINDINGS:")
print(f"   Most important predictors: {', '.join(feature_importance['Feature'].head(3).tolist())}")
print(f"   Model performs well for typical consumption patterns")
print(f"   Cyclical time features effectively capture daily/weekly patterns")

print(f"\n3. FORECAST INSIGHTS (Next 24h):")
future_hourly = future_forecast.groupby('hour')['predicted_power'].mean()
peak_hour_future = future_hourly.idxmax()
min_hour_future = future_hourly.idxmin()
print(f"   Expected peak demand at {peak_hour_future}:00 ({future_hourly.max():.2f} kW)")
print(f"   Expected minimum demand at {min_hour_future}:00 ({future_hourly.min():.2f} kW)")
print(f"   Average forecast: {future_predictions.mean():.2f} kW")


