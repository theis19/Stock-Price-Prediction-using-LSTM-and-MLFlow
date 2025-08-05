import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ta.momentum import RSIIndicator
from ta.trend import MACD
import mlflow

# Import our MLflow utilities
from mlflow_utils import (
    setup_mlflow_experiment, log_data_parameters, log_split_parameters,
    log_evaluation_metrics, create_and_log_plots, log_preprocessing_artifacts,
    log_model_summary, log_training_metrics, log_tensorflow_model_with_artifacts,
    register_best_model, end_mlflow_run, print_mlflow_ui_info
)

def download_stock_data(tickers, years_back=10):
    """Download stock data and log data parameters"""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years_back)
    
    data = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    data = data.dropna()
    
    # Log data parameters using utility function
    log_data_parameters(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), len(data))
    
    return data

def create_features(data):
    """Create features from stock data"""
    df = pd.DataFrame(index=data['Close'].index)
    df['BMRI_Close'] = data['Close']['BMRI.JK']
    #df['Volume'] = data['Volume']['BMRI.JK']
    #df['RSI'] = RSIIndicator(close=df['BMRI_Close'], window=14).rsi()
    #df['MACD'] = MACD(close=df['BMRI_Close']).macd()
    #df['MA5'] = df['BMRI_Close'].rolling(window=5).mean()
    #df['MA20'] = df['BMRI_Close'].rolling(window=20).mean()
    #df['BBCA_Return'] = data['Close']['BBCA.JK'].pct_change()
    #df['BBRI_Return'] = data['Close']['BBCA.JK'].pct_change()
    #df['BBNI_Return'] = data['Close']['BBCA.JK'].pct_change()
    df.dropna(inplace=True)
    
    # Log feature information
    feature_names = df.columns.tolist()
    mlflow.log_param("feature_names", feature_names)
    mlflow.log_param("n_features", len(feature_names))
    mlflow.log_param("feature_engineering_complete", len(df))
    
    return df

def split_and_scale_data(df, train_ratio=0.8):
    """Split data into train/test and scale features"""
    split_index = int(len(df) * train_ratio)
    
    train_df = df[:split_index]
    test_df = df[split_index:]
    
    # Log split information using utility function
    log_split_parameters(train_ratio, len(train_df), len(test_df))
    
    # Scale features
    scalers = {}
    train_scaled = pd.DataFrame(index=train_df.index)
    test_scaled = pd.DataFrame(index=test_df.index)
    
    for col in df.columns:
        scaler = MinMaxScaler()
        train_scaled[col] = scaler.fit_transform(train_df[[col]]).flatten()
        test_scaled[col] = scaler.transform(test_df[[col]]).flatten()
        scalers[col] = scaler
    
    return train_scaled, test_scaled, scalers

def create_sequences(data, window_size, target_col='BMRI_Close', data_type=""):
    """Create sequences for LSTM training"""
    x, y = [], []
    for i in range(window_size, len(data)):
        x.append(data.iloc[i - window_size:i].values)
        y.append(data.iloc[i][target_col])
    
    X, y = np.array(x), np.array(y)
    
    # Log sequence information with data_type prefix to avoid conflicts
    if data_type:
        mlflow.log_param(f"{data_type}_X_shape", str(X.shape))
        mlflow.log_param(f"{data_type}_y_shape", str(y.shape))
    
    return X, y

def build_lstm_model(input_shape):
    """Build and compile LSTM model"""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['root_mean_squared_error'])
    
    # Log model parameters
    mlflow.log_param("total_params", model.count_params())
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("loss_function", "mean_squared_error")
    
    return model

def train_model(model, X_train, y_train, epochs=200, batch_size=64):
    """Train the model and log training parameters"""
    patience = 15
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    # Log training parameters
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("validation_split", 0.1)
    mlflow.log_param("early_stopping_patience", patience)
    mlflow.log_param("early_stopping_monitor", "val_loss")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    return history

def make_predictions_and_evaluate(model, X_test, y_test, scalers):
    """Make predictions and evaluate model performance"""
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform predictions
    target_scaler = scalers['BMRI_Close']
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)
    mae_percentage = (mae / np.mean(y_test_actual)) * 100
    
    # Print results
    print(f"MAE  : {mae:.2f} IDR")
    print(f"RMSE : {rmse:.2f} IDR")
    print(f"R²   : {r2:.4f}")
    print("Mean absolute error on test set: {:.2f}%".format(mae_percentage))
    
    return y_test_actual, y_pred, mae, rmse, r2, mae_percentage

def main():
    """Main execution function"""
    # Setup MLflow using utility function
    setup_mlflow_experiment("BMRI_Stock_Prediction", "BMRI_LSTM_Baseline")
    
    try:
        # === STEP 1: Download Data ===
        tickers = ['BMRI.JK']
        data = download_stock_data(tickers, years_back=10)
        
        # === STEP 2: Feature Engineering ===
        df = create_features(data)
        
        # === STEP 3: Train/Test Split and Scaling ===
        train_scaled, test_scaled, scalers = split_and_scale_data(df)
        log_preprocessing_artifacts(scalers, df.columns.tolist())
        
        # === STEP 4: Create Sequences ===
        window_size = 30
        X_train, y_train = create_sequences(train_scaled, window_size, data_type='train')
        X_test, y_test = create_sequences(test_scaled, window_size, data_type='test')
        
        # === STEP 5: Build Model ===
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        log_model_summary(model)
        
        # === STEP 6: Train Model ===
        history = train_model(model, X_train, y_train)
        log_training_metrics(history)
        
        # === STEP 7: Evaluate Model ===
        y_test_actual, y_pred, mae, rmse, r2, mae_percentage = make_predictions_and_evaluate(
            model, X_test, y_test, scalers
        )
        log_evaluation_metrics(mae, rmse, r2, mae_percentage, y_test_actual)
        
        # === STEP 8: Create Plots ===
        create_and_log_plots(y_test_actual, y_pred)
        
        # === STEP 9: Log Model Artifacts ===
        log_tensorflow_model_with_artifacts(model, X_test, "BMRI_LSTM_Model")
        
        # === STEP 10: Register Best Model ===
        register_best_model(rmse, "BMRI_LSTM_Best_Model")
        
        print_mlflow_ui_info()
        
    finally:
        # Always end MLflow run
        end_mlflow_run()

if __name__ == "__main__":
    main()