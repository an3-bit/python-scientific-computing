# Time Series Forecasting Analysis

A comprehensive time series forecasting project comparing multiple methodologies including Exponential Smoothing, Prophet, SARIMA, and VAR models on sales data.

## 📊 Project Overview

This repository contains a complete time series forecasting analysis implemented for academic research, focusing on sales data forecasting using various statistical and machine learning approaches. The project systematically compares different forecasting methodologies and provides detailed performance analysis.

### Dataset
- **Source**: `2025_T2_A2_Sales_Dynamic.xlsx`
- **Observations**: 242 monthly data points (2005-2025)
- **Type**: Univariate time series (sales data)
- **Frequency**: Monthly observations with clear seasonal patterns

## 🎯 Analysis Questions

### Question 1: Exponential Smoothing & Holt-Winters Method (3 marks)
**File**: `question1_exponential_smoothing.py`

Implementation of:
- Simple Exponential Smoothing (SES)
- Holt-Winters Triple Exponential Smoothing
- One-year ahead forecasting
- MASE calculation for model evaluation

**Results**:
- SES RMSE: 137.84
- Holt-Winters RMSE: 79.73
- Generated visualization: `question1_exponential_smoothing_analysis.png`

### Question 2: Prophet and Correlogram (5 marks)
**Files**: 
- `question2_prophet.py` - Main Prophet analysis
- `additive_remainder_analysis.py` - Dedicated remainder series analysis
- `tableau_comparison.py` - Validation against Tableau results

Implementation of:
- Prophet with additive seasonality
- Prophet with multiplicative seasonality
- ACF/PACF analysis of remainder series
- Component decomposition analysis
- Tableau validation and comparison

**Results**:
- Additive Prophet RMSE: 65.77 🏆 (Best Performance)
- Multiplicative Prophet RMSE: 66.24
- Generated visualizations: Multiple component and ACF/PACF plots

### Question 3: Seasonal ARIMA Modelling (4 marks)
**Files**:
- `question3_sarima_additive.py` - Main SARIMA analysis aligned with additive findings
- `question3_arima.py` - Alternative ARIMA implementation

Implementation of:
- Stationarity testing (ADF test)
- SARIMA model selection via grid search
- Model diagnostics and validation
- Residual analysis
- One-year ahead forecasting with prediction intervals

**Results**:
- Selected Model: SARIMA(1, 1, 1) × (0, 1, 1, 12)
- SARIMA RMSE: 87.13
- AIC: 2385.16
- Generated visualization: `question3_sarima_additive_analysis.png`

### Question 4: VAR Models (6 marks)
**File**: `question4_var.py`

Implementation of:
- Vector Autoregression (VAR) model discussion
- Multivariate time series analysis concepts
- Model selection criteria
- Theoretical framework for business applications

## 🏆 Model Performance Comparison

| Rank | Method | RMSE | Performance Gap |
|------|--------|------|----------------|
| 1st 🥇 | Prophet Additive | 65.77 | Baseline |
| 2nd 🥈 | Holt-Winters | 79.73 | +21.2% |
| 3rd 🥉 | SARIMA | 87.13 | +32.5% |
| 4th | Simple Exp. Smoothing | 137.84 | +109.6% |

## 🛠 Technical Stack

### Core Libraries
- **Python**: 3.13
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Visualization
- **Prophet**: Facebook's time series forecasting tool
- **statsmodels**: Statistical modeling and econometrics
- **scikit-learn**: Machine learning utilities
- **openpyxl**: Excel file handling

### Environment Management
- **Virtual Environment**: `myenv/` (included)
- **Activation**: `source myenv/bin/activate`

## 🚀 Getting Started

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/an3-bit/python-scientific-computing.git
cd forecasting_project
```

### Environment Setup
```bash
# Activate virtual environment
source myenv/bin/activate

# Verify installation
python --version
pip list
```

### Running the Analysis

#### Individual Questions
```bash
# Question 1: Exponential Smoothing
python question1_exponential_smoothing.py

# Question 2: Prophet Analysis
python question2_prophet.py

# Question 3: SARIMA Analysis
python question3_sarima_additive.py

# Question 4: VAR Models
python question4_var.py
```

#### Comprehensive Analysis
```bash
# Run complete analysis with model comparison
python comprehensive_analysis.py
```

## 📈 Key Findings

### Seasonal Patterns
- **Seasonality Type**: Additive (confirmed across multiple methods)
- **Seasonal Period**: 12 months (annual patterns)
- **Pattern Consistency**: Strong seasonal effects validated by multiple models

### Model Recommendations

1. **Best Overall**: **Prophet Additive Model**
   - Lowest RMSE (65.77)
   - Automatic seasonality detection
   - Business interpretability
   - Robust to outliers

2. **Traditional Alternative**: **Holt-Winters**
   - Good performance (RMSE: 79.73)
   - Simple implementation
   - Well-established methodology

3. **Statistical Rigor**: **SARIMA**
   - Statistical significance testing
   - Classical econometric approach
   - Parameter interpretability

### Business Implications
- Additive seasonality confirms consistent seasonal patterns
- Prophet's superior performance suggests complex seasonal interactions
- All models successfully capture 12-month seasonal cycles
- Forecasting horizon: Reliable for 12-month ahead predictions

## 📁 File Structure

```
forecasting_project/
├── README.md                           # This file
├── 2025_T2_A2_Sales_Dynamic.xlsx     # Original dataset
├── myenv/                             # Python virtual environment
│
├── Core Analysis Files:
├── question1_exponential_smoothing.py    # SES & Holt-Winters
├── question2_prophet.py                  # Prophet analysis
├── question3_sarima_additive.py          # SARIMA modeling
├── question4_var.py                      # VAR models
│
├── Supporting Analysis:
├── additive_remainder_analysis.py        # Prophet remainder analysis
├── tableau_comparison.py                 # Tableau validation
├── comprehensive_analysis.py             # Complete comparison
├── final_comparison.py                   # Final model ranking
│
├── Generated Visualizations:
├── question1_exponential_smoothing_analysis.png
├── question2_prophet_additive_components.png
├── question3_sarima_additive_analysis.png
├── additive_prophet_remainder_acf_pacf.png
├── final_model_comparison.png
└── [Additional plots and analysis outputs]
```

## 📊 Visualizations Generated

- **Exponential Smoothing**: Model fits, residuals, and forecasts
- **Prophet Components**: Trend, seasonal, and remainder decomposition
- **ACF/PACF Plots**: Correlogram analysis for model identification
- **SARIMA Diagnostics**: Residual analysis and model validation
- **Model Comparison**: Performance metrics and ranking visualization

## 🔍 Methodology

### Evaluation Metrics
- **RMSE**: Root Mean Square Error (primary metric)
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **AIC/BIC**: Information criteria for model selection
- **Ljung-Box**: Residual autocorrelation testing

### Validation Approach
- **Train/Test Split**: 230 training observations, 12 test observations
- **Out-of-sample Testing**: Final year used for validation
- **Cross-method Validation**: Results verified across multiple approaches
- **External Validation**: Tableau comparison for Prophet models

## 🎓 Academic Context

This project demonstrates:
- Comprehensive time series methodology comparison
- Statistical modeling best practices
- Business forecasting applications
- Academic research standards in quantitative analysis
- Reproducible research methodology

## 🤝 Contributing

This is an academic research project. For questions or discussions about methodology:

1. Review the analysis files for detailed implementation
2. Check generated visualizations for results interpretation
3. Refer to model diagnostics for statistical validation

## 📝 License

This project is developed for academic research purposes. Please cite appropriately if using for educational or research applications.

---

**Last Updated**: September 2025  
**Python Version**: 3.13  
**Status**: ✅ Complete Analysis Available
