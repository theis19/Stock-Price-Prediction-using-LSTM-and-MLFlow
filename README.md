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

## Main Workflow
   1. Fetch historical stock data for BMRI.JK (10 years) using yfinance.
   2. Clean the data by removing missing values.
   3. Split the dataset into training (80%) and testing (20%) sets based on time.
   4. Normalize the training data using MinMaxScaler (to avoid data leakage).
   5. Generate input sequences using a 30-day time window.
   6. Train an LSTM model on the training sequences.
   7. Predict future prices using the trained model on the test set.
   8. Inverse transform predictions and true values back to actual price scale.
   9. Evaluate model performance using RMSE, MAE, and R², and compare it to the naive forecast baseline.

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
Naive Forecast            
![alt text](https://github.com/theis19/Stock-Price-Prediction-using-LSTM-and-MLFlow/blob/main/prediction_plot_naive.png "Naive Forecast") 
Closing Price
![alt text](https://github.com/theis19/Stock-Price-Prediction-using-LSTM-and-MLFlow/blob/main/prediction_plot_close100.png "Closing Price 100 Epochs")
Multiple Features
![alt text](https://github.com/theis19/Stock-Price-Prediction-using-LSTM-and-MLFlow/blob/main/prediction_plot_addfeature.png "Multiple Features")
Closing Price with 200 Epochs
![alt text](https://github.com/theis19/Stock-Price-Prediction-using-LSTM-and-MLFlow/blob/main/prediction_plot_close200.png "Closing Price 200 Epochs")


MLFlow Features Screenshot
![alt text](https://github.com/theis19/Stock-Price-Prediction-using-LSTM-and-MLFlow/blob/main/features.png "Features")

MLFlow Result Comparison Screenshot
![alt text](https://github.com/theis19/Stock-Price-Prediction-using-LSTM-and-MLFlow/blob/main/mlflowresult.png "Comparison Result")

## Result RMSE:
### Naive Forecast          : 107.8 IDR
### Closing Price 100 Epoch : 118.3 IDR
### Multiple Features       : 140.1 IDR
### Closing Price 200 Epoch : 113.2 IDR

## Things to Improve

This project focuses on building a working LSTM-based stock prediction model with basic experiment tracking. Given more time, several improvements could be made:

- **Hyperparameter Tuning**: Current settings were chosen manually. Optimization could improve results.
- **Model Comparison**: Other models like GRU, CNN, or XGBoost could be evaluated.
- **Cross-Validation**: The evaluation is based on a single split; time series cross-validation would be more robust.
- **Multi-step Prediction**: Currently only next-day forecasting is supported — multi-day prediction could be explored.
- **Feature Exploration**: Feature selection and more external features (e.g., market indicators) may enhance performance.

## How to Run

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
