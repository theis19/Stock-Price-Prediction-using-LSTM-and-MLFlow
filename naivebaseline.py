import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow

# Import our MLflow utilities
from mlflow_utils import (
    setup_mlflow_experiment, log_data_parameters, log_split_parameters,
    log_evaluation_metrics, create_and_log_plots, log_simple_model_artifact,
    end_mlflow_run, print_mlflow_ui_info
)

def download_stock_data(ticker='BMRI.JK', years_back=10):
    """Download BMRI stock data"""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years_back)
    
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    data = data.dropna()
    
    # Log data parameters using utility function
    log_data_parameters(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), len(data))
    
    return data

def prepare_naive_forecast_data(data, train_ratio=0.8):
    """Prepare data for naive forecasting"""
    # Use only closing prices
    prices = data['Close'].values
    
    # Split data
    split_index = int(len(prices) * train_ratio)
    train_prices = prices[:split_index]
    test_prices = prices[split_index:]
    
    # For naive forecast: prediction = previous day's price
    # So we need prices[:-1] as features and prices[1:] as targets
    test_actual = test_prices[1:]  # actual prices we want to predict
    test_predictions = test_prices[:-1]  # naive predictions (previous day's price)
    
    # Log model and split information
    mlflow.log_param("model_type", "Naive Forecast")
    mlflow.log_param("prediction_method", "Next day price = Current day price")
    log_split_parameters(train_ratio, len(train_prices), len(test_actual))
    
    return test_actual, test_predictions

def evaluate_naive_model(y_actual, y_pred):
    """Evaluate naive forecast performance"""
    # Calculate metrics
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
    mae_percentage = (mae / np.mean(y_actual)) * 100
    
    # Print results
    print(f"=== NAIVE FORECAST RESULTS ===")
    print(f"MAE  : {mae:.2f} IDR")
    print(f"RMSE : {rmse:.2f} IDR")
    print(f"R²   : {r2:.4f}")
    print("Mean absolute error on test set: {:.2f}%".format(mae_percentage))
    
    return mae, rmse, r2, mae_percentage

def main():
    """Main execution function"""
    # Setup MLflow using utility function
    setup_mlflow_experiment("BMRI_Stock_Prediction", "BMRI_Naive_Forecast")
    
    try:
        print("🔄 Starting Naive Forecast for BMRI...")
        
        # === STEP 1: Download Data ===
        data = download_stock_data('BMRI.JK', years_back=10)
        
        # === STEP 2: Prepare Naive Forecast ===
        y_actual, y_pred = prepare_naive_forecast_data(data)
        
        # === STEP 3: Evaluate Model ===
        mae, rmse, r2, mae_percentage = evaluate_naive_model(y_actual, y_pred)
        log_evaluation_metrics(mae, rmse, r2, mae_percentage, y_actual)
        
        # === STEP 4: Create Plots ===
        create_and_log_plots(y_actual, y_pred, "Naive Forecast - ", "Naive Forecast (Previous Day)")
        
        # === STEP 5: Save Model Artifact ===
        model_info = {
            "model_type": "Naive Forecast",
            "description": "Predicts next day's closing price as current day's closing price",
            "parameters": {
                "lag": 1,
                "prediction_rule": "price(t+1) = price(t)"
            }
        }
        log_simple_model_artifact(model_info, "naive_model.joblib")
        
        print(f"\n✅ Naive forecast completed!")
        print_mlflow_ui_info()
        
    finally:
        # Always end MLflow run
        end_mlflow_run()

if __name__ == "__main__":
    main()