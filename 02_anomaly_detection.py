import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    df = pd.read_csv('data/household_sample.csv')
    print(f"Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print("Error: household_sample.csv not found in data/ folder")
    exit()
df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                  format='%d/%m/%Y %H:%M:%S')
#Create time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Handle missing values
df_clean = df.fillna(method='ffill').fillna(method='bfill')

print(f"📅 Date range: {df_clean['timestamp'].min()} to {df_clean['timestamp'].max()}")
print(f"⏱️  Duration: {(df_clean['timestamp'].max() - df_clean['timestamp'].min()).days} days")

# STEP 2: FEATURE ENGINEERING FOR ANOMALY DETECTION
# Create features for anomaly detection
features_df = pd.DataFrame()

# Power metrics
features_df['global_active_power'] = df_clean['Global_active_power']
features_df['global_reactive_power'] = df_clean['Global_reactive_power']
features_df['voltage'] = df_clean['Voltage']
features_df['global_intensity'] = df_clean['Global_intensity']

# Sub-metering
features_df['sub_metering_1'] = df_clean['Sub_metering_1']
features_df['sub_metering_2'] = df_clean['Sub_metering_2']
features_df['sub_metering_3'] = df_clean['Sub_metering_3']

# Total sub-metered power
features_df['total_submetering'] = (df_clean['Sub_metering_1'] + 
                                     df_clean['Sub_metering_2'] + 
                                     df_clean['Sub_metering_3'])

# Unmetered power (potential theft indicator)
# Convert Global_active_power from kW to Wh: kW * 1000 / 60 (since data is per minute)
features_df['unmetered_power'] = (df_clean['Global_active_power'] * 1000 / 60) - features_df['total_submetering']

# Time features
features_df['hour'] = df_clean['hour']
features_df['is_weekend'] = df_clean['is_weekend']

# Power factor (indicator of equipment efficiency)
features_df['power_factor'] = features_df['global_active_power'] / (
    np.sqrt(features_df['global_active_power']**2 + features_df['global_reactive_power']**2) + 1e-6
)

print(f"Created {features_df.shape[1]} features for anomaly detection")
print(f"\nFeatures: {features_df.columns.tolist()}")

#STEP 3: TRAIN ISOLATION FOREST MODEL
# ============================================================================
print("\nSTEP 3: Training Anomaly Detection Model...")

# Select features for model
model_features = [
    'global_active_power', 'global_reactive_power', 'voltage', 
    'global_intensity', 'total_submetering', 'unmetered_power', 
    'power_factor', 'hour'
]

X = features_df[model_features].copy()

# Handle any remaining infinite or NaN values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Training data shape: {X_scaled.shape}")

# Train Isolation Forest
# contamination: expected proportion of outliers (0.05 = 5%)
iso_forest = IsolationForest(
    contamination=0.05,
    random_state=42,
    n_estimators=100,
    max_samples='auto',
    verbose=0
)

print("Training Isolation Forest...")
predictions = iso_forest.fit_predict(X_scaled)

# Get anomaly scores (lower is more anomalous)
anomaly_scores = iso_forest.score_samples(X_scaled)

# Add predictions to dataframe
features_df['anomaly'] = predictions
features_df['anomaly_score'] = anomaly_scores
features_df['timestamp'] = df_clean['timestamp']

# -1 indicates anomaly, 1 indicates normal
features_df['is_anomaly'] = (features_df['anomaly'] == -1).astype(int)

print(f"Model trained successfully!")

# STEP 4: ANALYZE RESULTS

print("\n STEP 4: Analyzing Results...")

total_anomalies = features_df['is_anomaly'].sum()
anomaly_percentage = (total_anomalies / len(features_df)) * 100

print(f"\n ANOMALY DETECTION SUMMARY:")
print(f"   • Total data points: {len(features_df):,}")
print(f"   • Anomalies detected: {total_anomalies:,}")
print(f"   • Anomaly rate: {anomaly_percentage:.2f}%")

# Separate anomalies
anomalies = features_df[features_df['is_anomaly'] == 1].copy()
normal = features_df[features_df['is_anomaly'] == 0].copy()

print(f"\n NORMAL vs ANOMALOUS CONSUMPTION:")
print(f"   Normal - Mean: {normal['global_active_power'].mean():.3f} kW")
print(f"   Anomaly - Mean: {anomalies['global_active_power'].mean():.3f} kW")
print(f"   Difference: {abs(anomalies['global_active_power'].mean() - normal['global_active_power'].mean()):.3f} kW")

# Analyze anomaly types
print(f"\n ANOMALY CHARACTERISTICS:")

# High consumption anomalies
high_consumption = anomalies[anomalies['global_active_power'] > normal['global_active_power'].quantile(0.95)]
print(f"   • High consumption anomalies: {len(high_consumption):,} ({len(high_consumption)/total_anomalies*100:.1f}%)")

# Low consumption anomalies
low_consumption = anomalies[anomalies['global_active_power'] < normal['global_active_power'].quantile(0.05)]
print(f"   • Low consumption anomalies: {len(low_consumption):,} ({len(low_consumption)/total_anomalies*100:.1f}%)")

# Voltage anomalies
voltage_anomalies = anomalies[
    (anomalies['voltage'] > normal['voltage'].quantile(0.95)) | 
    (anomalies['voltage'] < normal['voltage'].quantile(0.05))
]
print(f"   • Voltage anomalies: {len(voltage_anomalies):,} ({len(voltage_anomalies)/total_anomalies*100:.1f}%)")

# Unmetered power anomalies (potential theft)
theft_indicators = anomalies[anomalies['unmetered_power'] > normal['unmetered_power'].quantile(0.95)]
print(f"   • Potential theft indicators: {len(theft_indicators):,} ({len(theft_indicators)/total_anomalies*100:.1f}%)")


# STEP 5: TEMPORAL ANALYSIS OF ANOMALIES
print(f"\n TEMPORAL PATTERNS:")

anomalies['hour'] = anomalies['timestamp'].dt.hour
hourly_anomalies = anomalies.groupby('hour').size()

print(f"\n Peak Anomaly Hours:")
print(hourly_anomalies.nlargest(5).to_string())

# Weekday vs Weekend
weekday_anomalies = anomalies[anomalies['is_weekend'] == 0]
weekend_anomalies = anomalies[anomalies['is_weekend'] == 1]

print(f"\n Weekday vs Weekend Anomalies:")
print(f"   • Weekday: {len(weekday_anomalies):,} ({len(weekday_anomalies)/len(anomalies)*100:.1f}%)")
print(f"   • Weekend: {len(weekend_anomalies):,} ({len(weekend_anomalies)/len(anomalies)*100:.1f}%)")


# STEP 6: VISUALIZATIONS
print(f"\n STEP 6: Creating Visualizations...")

plt.style.use('seaborn-v0_8-darkgrid')

# Figure 1: Anomaly Score Distribution
print("   Creating Figure 1: Anomaly score distribution...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(normal['anomaly_score'], bins=50, alpha=0.7, label='Normal', color='green', edgecolor='black')
ax.hist(anomalies['anomaly_score'], bins=50, alpha=0.7, label='Anomaly', color='red', edgecolor='black')
ax.set_xlabel('Anomaly Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Anomaly Scores', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 2: Power Consumption - Normal vs Anomaly
print("   Creating Figure 2: Power consumption comparison...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(normal['global_active_power'], bins=50, alpha=0.6, label='Normal', color='blue', edgecolor='black')
ax.hist(anomalies['global_active_power'], bins=50, alpha=0.6, label='Anomaly', color='red', edgecolor='black')
ax.set_xlabel('Global Active Power (kW)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Power Consumption: Normal vs Anomalous', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 3: Anomalies Over Time
print("   Creating Figure 3: Anomalies timeline...")
fig, ax = plt.subplots(figsize=(15, 6))

# Sample data for visualization (plot every 10th point for clarity)
sample_normal = normal.iloc[::10]
sample_anomalies = anomalies.iloc[::10]

ax.scatter(sample_normal['timestamp'], sample_normal['global_active_power'], 
           c='blue', alpha=0.3, s=10, label='Normal')
ax.scatter(sample_anomalies['timestamp'], sample_anomalies['global_active_power'], 
           c='red', alpha=0.8, s=30, label='Anomaly', marker='x')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Power (kW)', fontsize=12)
ax.set_title('Anomaly Detection Timeline', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Figure 4: Hourly Anomaly Distribution
print("   Creating Figure 4: Hourly anomaly distribution...")
fig, ax = plt.subplots(figsize=(12, 6))
hourly_anomalies.plot(kind='bar', ax=ax, color='crimson', edgecolor='black', alpha=0.7)
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Number of Anomalies', fontsize=12)
ax.set_title('Anomaly Distribution by Hour', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Figure 5: Feature Importance Visualization
print("   Creating Figure 5: Anomaly characteristics...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Power comparison
axes[0, 0].boxplot([normal['global_active_power'], anomalies['global_active_power']], 
                    labels=['Normal', 'Anomaly'])
axes[0, 0].set_ylabel('Power (kW)', fontsize=11)
axes[0, 0].set_title('Active Power Distribution', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Voltage comparison
axes[0, 1].boxplot([normal['voltage'], anomalies['voltage']], 
                    labels=['Normal', 'Anomaly'])
axes[0, 1].set_ylabel('Voltage (V)', fontsize=11)
axes[0, 1].set_title('Voltage Distribution', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Unmetered power comparison
axes[1, 0].boxplot([normal['unmetered_power'], anomalies['unmetered_power']], 
                    labels=['Normal', 'Anomaly'])
axes[1, 0].set_ylabel('Unmetered Power (Wh)', fontsize=11)
axes[1, 0].set_title('Unmetered Power (Theft Indicator)', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Power factor comparison
axes[1, 1].boxplot([normal['power_factor'], anomalies['power_factor']], 
                    labels=['Normal', 'Anomaly'])
axes[1, 1].set_ylabel('Power Factor', fontsize=11)
axes[1, 1].set_title('Power Factor Distribution', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Anomaly Characteristics Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()


# STEP 7: SAVE RESULTS
# Save anomalies to CSV
anomalies_output = anomalies[[
    'timestamp', 'global_active_power', 'voltage', 'global_intensity',
    'unmetered_power', 'anomaly_score', 'hour', 'is_weekend'
]].copy()

anomalies_output.to_csv('data/detected_anomalies.csv', index=False)
print(f" Saved: data/detected_anomalies.csv ({len(anomalies_output):,} anomalies)")

# Save full dataset with anomaly flags
features_df.to_csv('data/energy_data_with_anomalies.csv', index=False)
print(f" Saved: data/energy_data_with_anomalies.csv ({len(features_df):,} records)")



# STEP 8: KEY INSIGHTS & RECOMMENDATIONS
print(f"\n1. ANOMALY DETECTION PERFORMANCE:")
print(f"    Detected {total_anomalies:,} anomalies ({anomaly_percentage:.2f}% of data)")
print(f"    Model used: Isolation Forest with 100 estimators")
print(f"    Contamination rate: 5%")

print(f"\n2. CRITICAL FINDINGS:")
if len(high_consumption) > 0:
    print(f"     {len(high_consumption):,} high consumption spikes detected")
    print(f"      → Could indicate faulty equipment or unauthorized usage")
if len(theft_indicators) > 0:
    print(f"     {len(theft_indicators):,} potential energy theft indicators")
    print(f"      → Large unmetered power suggests bypassed meters")
if len(voltage_anomalies) > 0:
    print(f"     { len(voltage_anomalies):,} voltage anomalies detected")
    print(f"      → May indicate grid instability or equipment issues")

print(f"\n3. TEMPORAL PATTERNS:")
peak_hour = hourly_anomalies.idxmax()
print(f"   • Most anomalies occur at hour {peak_hour}:00")
print(f"   • Weekday anomalies: {len(weekday_anomalies)/len(anomalies)*100:.1f}%")
print(f"   • Weekend anomalies: {len(weekend_anomalies)/len(anomalies)*100:.1f}%")
