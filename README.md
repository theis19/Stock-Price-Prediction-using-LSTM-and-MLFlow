# Stock Price Prediction Using LSTM + MLflow Tracking

This project implements a time series forecasting pipeline using an LSTM (Long Short-Term Memory) model to predict **BMRI.JK (Bank Mandiri)** stock prices. The pipeline includes feature engineering, model training, evaluation, and complete **MLflow integration** for experiment tracking, model management, and reproducibility.

## Objectives

- Predict the next-day stock price of BMRI.JK using historical data.
- Compare performance against a naive baseline model.
- Log all key metrics, artifacts, and model versions with MLflow.
- Register the best model automatically based on RMSE.

## Model Architecture

- 2 LSTM layers (50 units each)
- Dropout layers (0.2)
- Final Dense layer with 1 output neuron
- Optimizer: Adam (lr=0.001)
- Loss: Mean Squared Error (MSE)
- Metric: Root Mean Squared Error (RMSE)

## Features Used
   Main Feature:
   - **BMRI.JK Closing Price**

   Side Features (Not shown in the result because they worsen the model performance):
   - Volume
   - MACD
   - RSI
   - MA5
   - MA20
   - BBNI, BBCA, BBRI Price Change as a part of 4 Big Banks


## Project Structure

```
.
├── LSTM_script.py            # Main training script with MLflow integration
├── mlflow_utils.py           # Utility functions for logging and model management
├── naivebaseline.py          # Naive forecast script for baseline using today's close price to predict the next day's
├── mlruns/                   # MLflow run logs (created at runtime)
└── README.md                 # Project documentation (this file)
```

## Experiment Tracking with MLflow

Each run tracks:

- Parameters: tickers, window size, model architecture, train/test sizes, etc.
- Metrics: RMSE, MAE, R², MAE percentage, mean/STD of actual prices.
- Artifacts: 
  - Training curves (loss/RMSE)
  - Prediction vs. actual price plot
  - Residual histogram
  - Model summary
  - Trained `.h5` model + scaler + config
- Auto-registration of best model based on lowest RMSE.

Start the MLflow UI locally:
```bash
mlflow ui
```
Access at: [http://localhost:5000](http://localhost:5000)

## Baseline Comparison

A naive forecast (predicting next day's price as today’s) was used as a baseline. Results were compared using MAE and RMSE to assess the value of the LSTM model.

## Evaluation Metrics

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score**
- **MAE Percentage** (relative to mean actual price)

These metrics are logged and plotted after every experiment.

## Result
Naive Forecast            |  Closing Price
:-------------------------:|:-------------------------:
![](https://github.com/theis19/Stock-Price-Prediction-using-LSTM-and-MLFlow/blob/main/prediction_plot_naive.png)  |  ![]((https://github.com/theis19/Stock-Price-Prediction-using-LSTM-and-MLFlow/blob/main/prediction_plot_close100.png))

Multiple Features         |  Closing Price with 200 Epochs
:-------------------------:|:-------------------------:
![]((https://github.com/theis19/Stock-Price-Prediction-using-LSTM-and-MLFlow/blob/main/prediction_plot_addfeature.png))  |  ![](https://github.com/theis19/Stock-Price-Prediction-using-LSTM-and-MLFlow/blob/main/prediction_plot_close200.png)


## ✅ How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:
   ```bash
   python LSTM_script.py
   ```

3. View MLflow dashboard:
   ```bash
   mlflow ui
   ```
