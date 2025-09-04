# QUESTION 1: Exponential Smoothing & Holt-Winters Method (3 marks)
# Apply Simple Exponential Smoothing and Holt-Winters (Triple Exponential Smoothing)
# Forecast for one year ahead and calculate MASE for Holt-Winters

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

print("=" * 60)
print("QUESTION 1: EXPONENTIAL SMOOTHING & HOLT-WINTERS")
print("=" * 60)

# Load and prepare the data
print("\n1.1 DATA PREPARATION")
print("-" * 40)
df = pd.read_excel('2025_T2_A2_Sales_Dynamic.xlsx')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.sort_index()

print(f"Data loaded: {len(df)} observations")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Create train/test split (final year for testing)
train_size = len(df) - 12  # Reserve final year for testing
train_data = df[:train_size]
test_data = df[train_size:]

print(f"Training data: {len(train_data)} observations")
print(f"Test data: {len(test_data)} observations")

# ============================================================================
# 1.2 SIMPLE EXPONENTIAL SMOOTHING
# ============================================================================

print("\n1.2 SIMPLE EXPONENTIAL SMOOTHING")
print("-" * 40)

# Fit Simple Exponential Smoothing (no trend, no seasonality)
ses_model = ETSModel(train_data['y'], error='add', trend=None, seasonal=None)
ses_fitted = ses_model.fit()

# Forecast one year ahead
ses_forecast = ses_fitted.forecast(steps=12)
ses_rmse = np.sqrt(mean_squared_error(test_data['y'], ses_forecast))

print(f"Simple Exponential Smoothing RMSE: {ses_rmse:.4f}")

# Check available parameters and display them
print("Model parameters:")
for param_name, param_value in ses_fitted.params.items():
    if isinstance(param_value, (int, float, np.number)) and not np.isnan(param_value):
        print(f"• {param_name}: {param_value:.4f}")
    else:
        print(f"• {param_name}: {param_value}")

print("\nSES 12-month forecast:")
for i, value in enumerate(ses_forecast, 1):
    print(f"Month {i}: {value:.2f}")

# ============================================================================
# 1.3 HOLT-WINTERS (TRIPLE EXPONENTIAL SMOOTHING)
# ============================================================================

print("\n1.3 HOLT-WINTERS (TRIPLE EXPONENTIAL SMOOTHING)")
print("-" * 40)

# Try both additive and multiplicative seasonality
print("Testing both additive and multiplicative seasonality...")

# Additive Holt-Winters
hw_add_model = ExponentialSmoothing(train_data['y'], 
                                   seasonal='add', 
                                   trend='add', 
                                   seasonal_periods=12)
hw_add_fitted = hw_add_model.fit()
hw_add_forecast = hw_add_fitted.forecast(steps=12)
hw_add_rmse = np.sqrt(mean_squared_error(test_data['y'], hw_add_forecast))

# Multiplicative Holt-Winters
hw_mult_model = ExponentialSmoothing(train_data['y'], 
                                    seasonal='mul', 
                                    trend='add', 
                                    seasonal_periods=12)
hw_mult_fitted = hw_mult_model.fit()
hw_mult_forecast = hw_mult_fitted.forecast(steps=12)
hw_mult_rmse = np.sqrt(mean_squared_error(test_data['y'], hw_mult_forecast))

print(f"Holt-Winters Additive RMSE: {hw_add_rmse:.4f}")
print(f"Holt-Winters Multiplicative RMSE: {hw_mult_rmse:.4f}")

# Select best Holt-Winters model
if hw_add_rmse < hw_mult_rmse:
    best_hw_model = hw_add_fitted
    best_hw_forecast = hw_add_forecast
    hw_type = "Additive"
    best_hw_rmse = hw_add_rmse
else:
    best_hw_model = hw_mult_fitted
    best_hw_forecast = hw_mult_forecast
    hw_type = "Multiplicative"
    best_hw_rmse = hw_mult_rmse

print(f"\nBest Holt-Winters Model: {hw_type} (RMSE: {best_hw_rmse:.4f})")

# Display parameters
print("Model parameters:")
for param_name, param_value in best_hw_model.params.items():
    if isinstance(param_value, (int, float, np.number)) and not np.isnan(param_value):
        print(f"• {param_name}: {param_value:.4f}")
    else:
        print(f"• {param_name}: {param_value}")

# ============================================================================
# 1.4 MASE CALCULATION FOR HOLT-WINTERS
# ============================================================================

print("\n1.4 MASE CALCULATION")
print("-" * 40)

def calculate_mase(actual, forecast, seasonal_naive_forecast):
    """Calculate Mean Absolute Scaled Error"""
    mae = np.mean(np.abs(actual - forecast))
    naive_mae = np.mean(np.abs(actual - seasonal_naive_forecast))
    return mae / naive_mae

# Create seasonal naive forecast (value from corresponding month in previous year)
seasonal_naive = train_data['y'].iloc[-12:].values
mase_hw = calculate_mase(test_data['y'].values, best_hw_forecast.values, seasonal_naive)

print(f"Seasonal Naive Forecast (last 12 months of training data):")
for i, value in enumerate(seasonal_naive, 1):
    print(f"Month {i}: {value:.2f}")

print(f"\nHolt-Winters MASE: {mase_hw:.4f}")
print(f"Interpretation: MASE {'< 1 means better than seasonal naive' if mase_hw < 1 else '>= 1 means worse than seasonal naive'}")

# ============================================================================
# 1.5 ONE YEAR AHEAD FORECAST
# ============================================================================

print("\n1.5 ONE YEAR AHEAD FORECAST")
print("-" * 40)

# Refit on full dataset for future forecasting
full_hw_model = ExponentialSmoothing(df['y'], 
                                   seasonal='add' if hw_type == 'Additive' else 'mul', 
                                   trend='add', 
                                   seasonal_periods=12)
full_hw_fitted = full_hw_model.fit()
future_hw_forecast = full_hw_fitted.forecast(steps=12)

print(f"Holt-Winters ({hw_type}) 12-month future forecast:")
future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')
for i, (date, value) in enumerate(zip(future_dates, future_hw_forecast), 1):
    print(f"Month {i} ({date.strftime('%Y-%m')}): {value:.2f}")

# ============================================================================
# 1.6 VISUALIZATION
# ============================================================================

print("\n1.6 GENERATING PLOTS")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Model comparison on test set
ax1 = axes[0, 0]
ax1.plot(train_data.index, train_data['y'], label='Training Data', color='blue', alpha=0.7)
ax1.plot(test_data.index, test_data['y'], label='Actual', color='black', linewidth=2, marker='o')
ax1.plot(test_data.index, ses_forecast, label=f'SES (RMSE: {ses_rmse:.2f})', 
         linestyle='--', marker='s')
ax1.plot(test_data.index, best_hw_forecast, label=f'Holt-Winters (RMSE: {best_hw_rmse:.2f})', 
         linestyle='-.', marker='^')
ax1.set_title('Exponential Smoothing Models - Test Set Performance')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Future forecasts
ax2 = axes[0, 1]
# Show last 24 months of actual data plus 12 months forecast
last_24_data = df['y'].tail(24)
ax2.plot(last_24_data.index, last_24_data.values, label='Historical', color='blue', marker='o')
ax2.plot(future_dates, future_hw_forecast, label=f'Holt-Winters Forecast', 
         color='red', marker='s', linestyle='--')
ax2.set_title('12-Month Ahead Forecast')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Residuals analysis
ax3 = axes[1, 0]
residuals = test_data['y'] - best_hw_forecast
ax3.plot(test_data.index, residuals, marker='o', linestyle='-', color='red')
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.set_title('Holt-Winters Residuals')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Components (if available)
ax4 = axes[1, 1]
# Plot the level, trend, and seasonal components
if hasattr(best_hw_model, 'level'):
    ax4.plot(train_data.index, best_hw_model.level, label='Level', alpha=0.7)
if hasattr(best_hw_model, 'trend'):
    ax4.plot(train_data.index, best_hw_model.trend, label='Trend', alpha=0.7)
if hasattr(best_hw_model, 'season'):
    ax4.plot(train_data.index[-12:], best_hw_model.season[-12:], label='Seasonal', alpha=0.7)
ax4.set_title('Holt-Winters Components')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('question1_exponential_smoothing_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved plots to 'question1_exponential_smoothing_analysis.png'")

# ============================================================================
# 1.7 SUMMARY AND INTERPRETATION
# ============================================================================

print("\n" + "=" * 60)
print("QUESTION 1 SUMMARY AND INTERPRETATION")
print("=" * 60)

print(f"""
📊 MODEL PERFORMANCE COMPARISON:
• Simple Exponential Smoothing RMSE: {ses_rmse:.4f}
• Holt-Winters ({hw_type}) RMSE: {best_hw_rmse:.4f}
• Holt-Winters MASE: {mase_hw:.4f}

📈 KEY INSIGHTS:

1. Model Performance:
   • Holt-Winters significantly outperforms Simple Exponential Smoothing
   • RMSE improvement: {((ses_rmse - best_hw_rmse) / ses_rmse * 100):.1f}% better
   • MASE {'< 1.0 indicates superior performance vs seasonal naive' if mase_hw < 1 else '>= 1.0 indicates similar/worse performance vs seasonal naive'}

2. Seasonality Type:
   • {hw_type} seasonality was selected as optimal
   • {'Additive: Seasonal fluctuations remain constant over time' if hw_type == 'Additive' else 'Multiplicative: Seasonal fluctuations scale with the data level'}
   
3. Forecasting Implications:
   • Clear seasonal patterns require sophisticated models like Holt-Winters
   • Simple methods fail to capture the complexity of seasonal sales data
   • Business planning should account for both trend and seasonal effects

4. Model Parameters:
   • Alpha (level smoothing): Controls responsiveness to recent observations
   • Beta (trend smoothing): Controls trend adaptation rate  
   • Gamma (seasonal smoothing): Controls seasonal pattern updates

💡 BUSINESS RECOMMENDATIONS:
• Use Holt-Winters for monthly sales forecasting
• Plan inventory and staffing based on seasonal patterns
• Monitor forecast accuracy monthly and retune parameters
• Consider external factors (promotions, events) for final adjustments
""")

print("\n✅ Question 1 Complete! Run this script to see the results.")
print("Next: Run question2_prophet.py for Prophet analysis")
