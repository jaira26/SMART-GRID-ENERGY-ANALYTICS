"""
Smart Grid Energy Analytics - Data Exploration
Author: Jairaghavendra Sridhar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. LOAD DATA
try:
    df = pd.read_csv('data/household_sample.csv')
    print(f"Data loaded successfully!")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
except FileNotFoundError:
    print("Error: File not found!")
    print("Make sure 'household_sample.csv' is in the 'data' folder")
    exit()

# 2. INITIAL DATA INSPECTION
print(df.head())

print(df.columns.tolist())

print(df.dtypes)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

# 3. DATA CLEANING & PREPARATION
df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                  format='%d/%m/%Y %H:%M:%S')
print("Timestamp created!")

df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
print(f"📅 Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"⏱️  Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

# Handle missing values (forward fill for small gaps)
print("\n Handling missing values...")
df_clean = df.fillna(method='ffill').fillna(method='bfill')
print(f"Missing values handled. Remaining: {df_clean.isnull().sum().sum()}")


print("STEP 4: Statistical Summary")

print("\n Power Consumption Statistics:")
stats = df_clean['Global_active_power'].describe()
print(stats)

print(f"\n Key Metrics:")
print(f"   • Mean: {df_clean['Global_active_power'].mean():.3f} kW")
print(f"   • Median: {df_clean['Global_active_power'].median():.3f} kW")
print(f"   • Std Dev: {df_clean['Global_active_power'].std():.3f} kW")
print(f"   • Peak: {df_clean['Global_active_power'].max():.3f} kW")
print(f"   • Minimum: {df_clean['Global_active_power'].min():.3f} kW")

# 5. TEMPORAL PATTERNS
# Hourly pattern
print("\n Average Consumption by Hour:")
hourly_pattern = df_clean.groupby('hour')['Global_active_power'].mean()
print(hourly_pattern)

print("\n Peak Hours:")
print(hourly_pattern.nlargest(5))

print("\n Low Usage Hours:")
print(hourly_pattern.nsmallest(5))

# Weekday vs Weekend
weekday_avg = df_clean[df_clean['is_weekend'] == 0]['Global_active_power'].mean()
weekend_avg = df_clean[df_clean['is_weekend'] == 1]['Global_active_power'].mean()

print(f"\n Weekday vs Weekend:")
print(f"   • Weekday Average: {weekday_avg:.3f} kW")
print(f"   • Weekend Average: {weekend_avg:.3f} kW")
print(f"   • Difference: {abs(weekday_avg - weekend_avg):.3f} kW ({((weekend_avg/weekday_avg - 1)*100):.1f}%)")

# 6. ANOMALY DETECTION 
# Using 3-sigma rule
mean_power = df_clean['Global_active_power'].mean()
std_power = df_clean['Global_active_power'].std()
threshold_high = mean_power + (3 * std_power)
threshold_low = mean_power - (3 * std_power)

anomalies_high = df_clean[df_clean['Global_active_power'] > threshold_high]
anomalies_low = df_clean[(df_clean['Global_active_power'] < threshold_low) & 
                         (df_clean['Global_active_power'] > 0)]

print(f"Anomaly Detection Results:")
print(f"   • High consumption anomalies: {len(anomalies_high):,} ({(len(anomalies_high)/len(df_clean)*100):.2f}%)")
print(f"   • Low consumption anomalies: {len(anomalies_low):,} ({(len(anomalies_low)/len(df_clean)*100):.2f}%)")
print(f"   • Threshold High: {threshold_high:.3f} kW")
print(f"   • Threshold Low: {threshold_low:.3f} kW")

if len(anomalies_high) > 0:
    print(f"\nTop 5 High Consumption Anomalies:")
    top_anomalies = anomalies_high.nlargest(5, 'Global_active_power')[
        ['timestamp', 'Global_active_power', 'hour', 'day_of_week']
    ]
    print(top_anomalies.to_string())

# 7. SUB-METERING ANALYSIS
print("\n Average Energy by Appliance Category:")
print(f"   • Kitchen (Sub_metering_1): {df_clean['Sub_metering_1'].mean():.2f} Wh")
print(f"   • Laundry (Sub_metering_2): {df_clean['Sub_metering_2'].mean():.2f} Wh")
print(f"   • HVAC (Sub_metering_3): {df_clean['Sub_metering_3'].mean():.2f} Wh")

total_submetering = (df_clean['Sub_metering_1'].mean() + 
                     df_clean['Sub_metering_2'].mean() + 
                     df_clean['Sub_metering_3'].mean())
print(f"   • Total Sub-metered: {total_submetering:.2f} Wh")

# Percentage breakdown
kitchen_pct = (df_clean['Sub_metering_1'].mean() / total_submetering) * 100
laundry_pct = (df_clean['Sub_metering_2'].mean() / total_submetering) * 100
hvac_pct = (df_clean['Sub_metering_3'].mean() / total_submetering) * 100

print(f"\n Percentage Breakdown:")
print(f"   • Kitchen: {kitchen_pct:.1f}%")
print(f"   • Laundry: {laundry_pct:.1f}%")
print(f"   • HVAC: {hvac_pct:.1f}%")

# 8. VISUALIZATIONS
# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Figure 1: Daily Average Consumption
print("\n Creating Figure 1: Daily consumption trend...")
fig, ax = plt.subplots(figsize=(15, 5))
daily_avg = df_clean.groupby(df_clean['timestamp'].dt.date)['Global_active_power'].mean()
ax.plot(daily_avg.index, daily_avg.values, linewidth=1.5, color='steelblue')
ax.set_title('Daily Average Power Consumption', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Power (kW)', fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Figure 2: Hourly Pattern
print("\n Creating Figure 2: Hourly consumption pattern...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(hourly_pattern.index, hourly_pattern.values, color='coral', alpha=0.7, edgecolor='black')
ax.set_title('Average Power Consumption by Hour of Day', fontsize=16, fontweight='bold')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Average Power (kW)', fontsize=12)
ax.set_xticks(range(24))
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Figure 3: Weekday vs Weekend
print("\n Creating Figure 3: Weekday vs weekend comparison...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

weekday_data = df_clean[df_clean['is_weekend'] == 0]
weekday_hourly = weekday_data.groupby('hour')['Global_active_power'].mean()
axes[0].plot(weekday_hourly.index, weekday_hourly.values, 
             marker='o', linewidth=2, markersize=8, color='#2E86AB')
axes[0].set_title('Weekday Pattern', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Hour', fontsize=12)
axes[0].set_ylabel('Power (kW)', fontsize=12)
axes[0].grid(True, alpha=0.3)

weekend_data = df_clean[df_clean['is_weekend'] == 1]
weekend_hourly = weekend_data.groupby('hour')['Global_active_power'].mean()
axes[1].plot(weekend_hourly.index, weekend_hourly.values, 
             marker='o', linewidth=2, markersize=8, color='#A23B72')
axes[1].set_title('Weekend Pattern', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Hour', fontsize=12)
axes[1].set_ylabel('Power (kW)', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Figure 4: Sub-metering Pie Chart
print("\n Creating Figure 4: Sub-metering breakdown...")
fig, ax = plt.subplots(figsize=(10, 8))
labels = ['Kitchen', 'Laundry', 'HVAC']
sizes = [
    df_clean['Sub_metering_1'].mean(),
    df_clean['Sub_metering_2'].mean(),
    df_clean['Sub_metering_3'].mean()
]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
explode = (0.05, 0.05, 0.05)

ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
       shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax.set_title('Energy Consumption by Appliance Category', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()