# FINAL COMPARISON: All Models Side-by-Side
# Run this after completing Questions 1-3 to compare all models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

# Load data
df = pd.read_excel('2025_T2_A2_Sales_Dynamic.xlsx')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.sort_index()

# Train/test split
train_size = len(df) - 12
train_data = df[:train_size]
test_data = df[train_size:]

print(f"Comparing models on {len(test_data)} test observations")

# Store all results
results = {}

# Re-run best models from each question (simplified)
print("\nRe-fitting best models from each question...")

# 1. Best Exponential Smoothing (from Question 1)
print("• Holt-Winters...")
hw_model = ExponentialSmoothing(train_data['y'], seasonal='mul', trend='add', seasonal_periods=12)
hw_fitted = hw_model.fit()
hw_forecast = hw_fitted.forecast(steps=12)
hw_rmse = np.sqrt(mean_squared_error(test_data['y'], hw_forecast))
results['Holt-Winters'] = hw_rmse

# 2. Best Prophet (from Question 2)  
print("• Prophet...")
df_prophet = df.reset_index().rename(columns={'Date': 'ds', 'y': 'y'})
train_prophet = df_prophet[:train_size]
test_prophet = df_prophet[train_size:]

prophet_model = Prophet(seasonality_mode='multiplicative')
prophet_model.fit(train_prophet)
prophet_forecast = prophet_model.predict(test_prophet[['ds']])
prophet_rmse = np.sqrt(mean_squared_error(test_prophet['y'], prophet_forecast['yhat']))
results['Prophet'] = prophet_rmse

# 3. SARIMA (simplified - using common good parameters)
print("• SARIMA...")
try:
    sarima_model = ARIMA(train_data['y'], order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima_fitted = sarima_model.fit()
    sarima_forecast = sarima_fitted.forecast(steps=12)
    sarima_rmse = np.sqrt(mean_squared_error(test_data['y'], sarima_forecast))
    results['SARIMA'] = sarima_rmse
except:
    print("  SARIMA fitting failed, trying simpler model...")
    sarima_model = ARIMA(train_data['y'], order=(0,1,1), seasonal_order=(0,1,1,12))
    sarima_fitted = sarima_model.fit()
    sarima_forecast = sarima_fitted.forecast(steps=12)
    sarima_rmse = np.sqrt(mean_squared_error(test_data['y'], sarima_forecast))
    results['SARIMA'] = sarima_rmse

# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

print("\n" + "=" * 60)
print("COMPREHENSIVE MODEL COMPARISON")
print("=" * 60)

# Display results
print(f"\n📊 MODEL PERFORMANCE (RMSE on Test Set):")
print("-" * 50)
for model, rmse in sorted(results.items(), key=lambda x: x[1]):
    print(f"• {model:<20}: {rmse:.4f}")

best_model_name = min(results.items(), key=lambda x: x[1])[0]
best_rmse = min(results.values())

print(f"\n🏆 BEST PERFORMING MODEL: {best_model_name} (RMSE: {best_rmse:.4f})")

# Calculate improvement percentages
print(f"\n📈 PERFORMANCE IMPROVEMENTS:")
baseline_rmse = max(results.values())
for model, rmse in sorted(results.items(), key=lambda x: x[1]):
    improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
    print(f"• {model}: {improvement:.1f}% better than worst model")

# ============================================================================
# VISUALIZATION
# ============================================================================

print(f"\n📊 GENERATING COMPARISON VISUALIZATION")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: All forecasts on test set
ax1 = axes[0, 0]
ax1.plot(train_data.index[-24:], train_data['y'].iloc[-24:], label='Training Data', 
         color='blue', alpha=0.7)
ax1.plot(test_data.index, test_data['y'], label='Actual Test Data', 
         color='black', linewidth=3, marker='o', markersize=6)
ax1.plot(test_data.index, hw_forecast, label=f'Holt-Winters ({hw_rmse:.2f})', 
         linestyle='--', marker='s', markersize=4)
ax1.plot(test_prophet['ds'], prophet_forecast['yhat'], label=f'Prophet ({prophet_rmse:.2f})', 
         linestyle='-.', marker='^', markersize=4)
ax1.plot(test_data.index, sarima_forecast, label=f'SARIMA ({sarima_rmse:.2f})', 
         linestyle=':', marker='d', markersize=4)
ax1.set_title('Model Comparison - Test Set Performance', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Plot 2: RMSE Comparison Bar Chart
ax2 = axes[0, 1]
models = list(results.keys())
rmses = list(results.values())
colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
bars = ax2.bar(models, rmses, color=colors)
ax2.set_title('RMSE Comparison Across Models', fontsize=14, fontweight='bold')
ax2.set_ylabel('RMSE')
ax2.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for bar, rmse in zip(bars, rmses):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{rmse:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 3: Residuals comparison
ax3 = axes[1, 0]
hw_residuals = test_data['y'] - hw_forecast
prophet_residuals = test_prophet['y'].values - prophet_forecast['yhat'].values  # Ensure same length
sarima_residuals = test_data['y'] - sarima_forecast

ax3.plot(test_data.index, hw_residuals, label='Holt-Winters', marker='o', alpha=0.7)
ax3.plot(test_data.index, prophet_residuals, label='Prophet', marker='s', alpha=0.7)
ax3.plot(test_data.index, sarima_residuals, label='SARIMA', marker='^', alpha=0.7)
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax3.set_title('Residuals Comparison', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Future forecasts (refit on full data)
ax4 = axes[1, 1]

# Refit models on full data for future forecasting
full_hw = ExponentialSmoothing(df['y'], seasonal='mul', trend='add', seasonal_periods=12).fit()
future_hw = full_hw.forecast(steps=12)

full_prophet = Prophet(seasonality_mode='multiplicative')
full_prophet.fit(df_prophet)
future_dates_prophet = full_prophet.make_future_dataframe(periods=12, freq='M')
future_prophet = full_prophet.predict(future_dates_prophet).tail(12)['yhat']

full_sarima = ARIMA(df['y'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
future_sarima = full_sarima.forecast(steps=12)

# Plot
future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')
ax4.plot(df.index[-12:], df['y'].iloc[-12:], label='Last 12 Months', 
         color='black', marker='o', linewidth=2)
ax4.plot(future_dates, future_hw, label='Holt-Winters', marker='s', linestyle='--')
ax4.plot(future_dates, future_prophet, label='Prophet', marker='^', linestyle='-.')
ax4.plot(future_dates, future_sarima, label='SARIMA', marker='d', linestyle=':')
ax4.set_title('12-Month Future Forecasts', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('final_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved comprehensive comparison to 'final_model_comparison.png'")

# ============================================================================
# FINAL RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 60)
print("FINAL RECOMMENDATIONS")
print("=" * 60)

print(f"""
🎯 MODEL SELECTION RECOMMENDATION:

Best Model: {best_model_name}
• Lowest RMSE: {best_rmse:.4f}
• Best balance of accuracy and interpretability

🔍 MODEL COMPARISON INSIGHTS:

Holt-Winters:
• Strengths: Simple, interpretable, handles trend and seasonality
• Weaknesses: Assumes constant relationships over time
• Best for: Stable seasonal patterns, operational planning

Prophet:  
• Strengths: Robust to outliers, handles holidays, uncertainty intervals
• Weaknesses: Less interpretable parameters, can overfit
• Best for: Irregular patterns, external event impacts

SARIMA:
• Strengths: Theoretical foundation, flexible, diagnostic tools
• Weaknesses: Complex parameter selection, assumes stationarity  
• Best for: Stationary series, when understanding dynamics is crucial

💼 BUSINESS RECOMMENDATIONS:

1. Primary Model: Use {best_model_name} for operational forecasting
2. Ensemble Approach: Combine top 2-3 models for robust predictions
3. Monitoring: Track forecast accuracy monthly, retune quarterly
4. Extensions: Collect additional data for VAR implementation
5. Validation: Use rolling forecast validation for model selection

📋 IMPLEMENTATION STRATEGY:
• Deploy {best_model_name} for immediate forecasting needs
• Develop ensemble methodology for critical business decisions
• Establish data collection for multivariate VAR models
• Create monitoring dashboard for ongoing forecast evaluation
""")

print(f"\n🎉 COMPLETE ANALYSIS FINISHED!")
print(f"Generated plots and analysis for all 4 questions.")
print(f"Best performing model: {best_model_name} with RMSE: {best_rmse:.4f}")
