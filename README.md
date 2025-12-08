# Food CPI Forecast with LSTM and Incidence Analysis

## Overview
This project develops an integrated framework to forecast Ecuador’s Food Consumer Price Index (CPI-Food) and diagnose the subgroups exerting the greatest inflationary pressure. It combines deep learning (LSTM) for short-term time-series forecasting with nonlinear regression (Random Forest) for incidence analysis.

**Author**: Mateo Pilaquinga  
**Institution**: Yachay Tech University, School of Mathematical and Computational Sciences  

## Objectives
1.  **Forecast**: Predict CPI-Food values for a 3-6 month horizon using a Long Short-Term Memory (LSTM) network.
2.  **Diagnose**: Identify key drivers of food inflation (e.g., bread, cereals, meat) using a Random Forest model.
3.  **Support Decision Making**: Provide tools for public institutions and private agents to anticipate inflation and design stabilization policies.

## Data Source
The dataset consists of official monthly series from the **National Institute of Statistics and Census (INEC)** of Ecuador.
-   **Period**: January 2006 to October 2025 (238 observations).
-   **Target Variable**: Food CPI (IPC-Alimentos).
-   **Features**: Eleven COICOP food subindices (e.g., Oils & Fats, Meat, Fruits, Vegetables).

## Methodology

### 1. Forecasting with LSTM
-   **Model**: Multi-step LSTM network.
-   **Input**: Sliding temporal windows of past observations (standardized).
-   **Output**: Joint estimation of future values (3-6 months ahead).
-   **Performance**: The LSTM achieves low error levels (RMSE = 1.18, MAPE = 1.01% on test set), capturing trends, seasonality, and shocks effectively.

### 2. Incidence Analysis with Random Forest
-   **Model**: Random Forest regressor trained on monthly variation rates.
-   **Goal**: Quantify the contribution of each food subgroup to the aggregate CPI variation.
-   **Findings**: Bread and cereals, non-alcoholic beverages, and meat are the primary structural drivers, while fresh products (fruits, vegetables) show higher volatility but marginal aggregate impact.

## Project Structure

```
├── data/
│   ├── processed/      # Processed data used for modeling (data.csv)
│   ├── ipc.csv         # Raw CPI data
│   └── productos.csv   # Raw product subindices data
├── src/                # Source code
│   ├── main_models.py          # Main entry point for the pipeline
│   ├── lstm_model.py           # LSTM model implementation
│   ├── random_forest_model.py  # Random Forest model implementation
│   ├── preprocessing.py        # Data cleaning and preparation modules
│   ├── evaluation.py           # Metrics and plotting utilities
│   └── compare_predictions.py  # Comparison script for validations
├── notebooks/          # Jupyter notebooks for exploration
│   ├── data_clean.ipynb        # Data cleaning process
│   ├── eda.ipynb               # Exploratory Data Analysis
│   └── future_eng.ipynb        # Feature engineering experiments
├── docs/               # Documentation
│   ├── clean_process.md        # Details on data cleaning
│   ├── description_variables.md # Variable dictionary
│   └── report_results.md       # Detailed model results
├── models/             # Directory for saved models (.h5, .pkl)
├── results/            # Generated metrics and predictions (.csv, .json)
├── figs/               # Generated plots and visualizations
└── requirements.txt    # Python dependencies
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/BrDenky/CFBPredic.git
    cd CFBPredic
    ```

2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the complete pipeline (preprocessing, training, evaluation, and visualization), execute the main script:

```bash
python src/main_models.py
```

This will:
1.  Load and preprocess the data from `data/processed/data.csv`.
2.  Train the LSTM model for forecasting.
3.  Train the Random Forest model for feature importance.
4.  Generate plots in `figs/` and save metrics in `results/`.
5.  Save trained models in `models/`.

## Results Summary

| Model | Metric | Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **LSTM** | RMSE | **1.18** | High accuracy, error magnitude close to index scale (~1 point) |
| **LSTM** | MAPE | **1.01%** | Excellent relative accuracy (<5%) |
| Random Forest | RMSE | 9.91 | Used for explanatory purposes (incidence), not forecasting |

**Key Inflation Drivers (Random Forest Importance)**:
1.  Bread and Cereals (~17%)
2.  Non-alcoholic Beverages (~16%)
3.  Meat (~14.6%)

## License
[MIT License](LICENSE) (or specify appropriate license)
