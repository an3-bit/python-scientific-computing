# Step 1: Import all the necessary libraries
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

print("--- Starting Forecast Analysis ---")

# Step 2: Load and prepare the data
# The Excel file should be in the same folder as this script
df = pd.read_excel('2025_T2_A2_Sales_Dynamic.xlsx')

# Prophet requires columns to be named 'ds' (datestamp) and 'y' (value)
df = df.rename(columns={'Date': 'ds', 'y': 'y'})
df['ds'] = pd.to_datetime(df['ds'])

print(f"Data loaded successfully with {len(df)} rows.")

# Step 3: Fit the Additive Prophet Model
print("\n--- Fitting Additive Model ---")
model_add = Prophet(seasonality_mode='additive')
model_add.fit(df)

# Make predictions on the historical data to evaluate the model
forecast_add = model_add.predict(df[['ds']])

# Calculate RMSE for the additive model
rmse_add = np.sqrt(mean_squared_error(df['y'], forecast_add['yhat']))
print(f"Additive Model RMSE: {rmse_add:.4f}")


# Step 4: Fit the Multiplicative Prophet Model
print("\n--- Fitting Multiplicative Model ---")
model_mult = Prophet(seasonality_mode='multiplicative')
model_mult.fit(df)

# Make predictions for evaluation
forecast_mult = model_mult.predict(df[['ds']])

# Calculate RMSE for the multiplicative model
rmse_mult = np.sqrt(mean_squared_error(df['y'], forecast_mult['yhat']))
print(f"Multiplicative Model RMSE: {rmse_mult:.4f}")


# Step 5: Compare models and select the best one
print("\n--- Model Comparison ---")
if rmse_add < rmse_mult:
    print(f"Additive model is better (RMSE: {rmse_add:.4f} < {rmse_mult:.4f}).")
    best_model = model_add
    best_forecast = forecast_add
else:
    print(f"Multiplicative model is better (RMSE: {rmse_mult:.4f} < {rmse_add:.4f}).")
    best_model = model_mult
    best_forecast = forecast_mult

# Step 6: Identify seasonality patterns and trends from the best model
# This will save a plot of the components to a file named 'forecast_components.png'
print("\nGenerating forecast components plot...")
fig_components = best_model.plot_components(best_forecast)
plt.savefig('forecast_components.png')
print("Saved 'forecast_components.png'.")


# Step 7: Obtain the Remainder (Residuals) series
# The remainder is the actual value (y) minus the forecasted value (yhat)
df['remainder'] = df['y'] - best_forecast['yhat']
print("\nCalculated the remainder series.")

# Step 8: Calculate and plot the ACF and PACF of the Remainder series
print("Generating ACF and PACF plots for the remainder series...")

fig_acf_pacf, axes = plt.subplots(2, 1, figsize=(10, 8))

# ACF Plot
plot_acf(df['remainder'], ax=axes[0], lags=40)
axes[0].set_title('Autocorrelation Function (ACF) of Remainder')

# PACF Plot
plot_pacf(df['remainder'], ax=axes[1], lags=40)
axes[1].set_title('Partial Autocorrelation Function (PACF) of Remainder')

plt.tight_layout()
plt.savefig('acf_pacf_plots.png')
print("Saved 'acf_pacf_plots.png'.")

print("\n--- Analysis Complete ---")