"""
MLflow utility functions for stock prediction experiments
Contains common MLflow functions used across different model scripts
"""

import mlflow
import mlflow.tensorflow
import mlflow.sklearn
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json
import tempfile
import os

def setup_mlflow_experiment(experiment_name="BMRI_Stock_Prediction", run_name=None):
    """Initialize MLflow tracking with experiment and run"""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)
    if run_name:
        mlflow.start_run(run_name=run_name)
    else:
        mlflow.start_run()

def log_data_parameters(ticker, start_date, end_date, total_points):
    """Log data download parameters"""
    if isinstance(ticker, list):
        mlflow.log_param("tickers", ticker)
    else:
        mlflow.log_param("ticker", ticker)
    mlflow.log_param("data_start_date", start_date)
    mlflow.log_param("data_end_date", end_date)
    mlflow.log_param("total_data_points", total_points)

def log_split_parameters(train_ratio, train_samples, test_samples):
    """Log train/test split parameters"""
    mlflow.log_param("train_test_split_ratio", train_ratio)
    mlflow.log_param("train_samples", train_samples)
    mlflow.log_param("test_samples", test_samples)

def log_evaluation_metrics(mae, rmse, r2, mae_percentage, y_actual):
    """Log evaluation metrics to MLflow"""
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_r2_score", r2)
    mlflow.log_metric("test_mae_percentage", mae_percentage)
    mlflow.log_metric("mean_actual_price", np.mean(y_actual))
    mlflow.log_metric("std_actual_price", np.std(y_actual))

def log_residual_statistics(residuals):
    """Log residual statistics"""
    mlflow.log_metric("residuals_mean", np.mean(residuals))
    mlflow.log_metric("residuals_std", np.std(residuals))
    mlflow.log_metric("residuals_min", np.min(residuals))
    mlflow.log_metric("residuals_max", np.max(residuals))

def create_and_log_prediction_plot(y_actual, y_pred, title_suffix="", model_label="Predicted"):
    """Create and log prediction plot"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual, label='Actual BMRI Price', alpha=0.8)
    plt.plot(y_pred, label=f'{model_label} BMRI Price', alpha=0.8)
    plt.title(f'BMRI Stock Price Prediction {title_suffix}(Test Set)')
    plt.xlabel('Time')
    plt.ylabel('Price (IDR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "prediction_plot.png")
    plt.show()

def create_and_log_residuals_plot(y_actual, y_pred, title_suffix=""):
    """Create and log residuals histogram"""
    residuals = y_actual.flatten() - y_pred.flatten()
    
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'{title_suffix}Prediction Residuals Distribution')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    mlflow.log_figure(plt.gcf(), "residuals_histogram.png")
    plt.close()
    
    # Log residual statistics
    log_residual_statistics(residuals)

def create_and_log_plots(y_actual, y_pred, title_suffix="", model_label="Predicted"):
    """Create and log both prediction and residuals plots"""
    create_and_log_prediction_plot(y_actual, y_pred, title_suffix, model_label)
    create_and_log_residuals_plot(y_actual, y_pred, title_suffix)

def log_simple_model_artifact(model_info, filename="model_info.joblib"):
    """Save and log a simple model artifact (for non-ML models like naive forecast)"""
    joblib.dump(model_info, filename)
    mlflow.log_artifact(filename)
    os.remove(filename)

def log_preprocessing_artifacts(scalers, feature_names, target_column="BMRI_Close"):
    """Log preprocessing artifacts (scalers and feature config)"""
    # Save scalers
    scaler_path = "scalers.joblib"
    joblib.dump(scalers, scaler_path)
    mlflow.log_artifact(scaler_path)
    os.remove(scaler_path)
    
    # Save feature config
    feature_config = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "target_column": target_column
    }
    with open("feature_config.json", "w") as f:
        json.dump(feature_config, f, indent=2)
    mlflow.log_artifact("feature_config.json")
    os.remove("feature_config.json")

def log_model_summary(model):
    """Log model architecture summary"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.flush()
    mlflow.log_artifact(f.name, "model_summary.txt")
    os.unlink(f.name)

def log_training_metrics(history):
    """Log training metrics and curves for neural networks"""
    # Log final metrics
    mlflow.log_metric("final_train_loss", history.history['loss'][-1])
    mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
    mlflow.log_metric("final_train_rmse", history.history['root_mean_squared_error'][-1])
    mlflow.log_metric("final_val_rmse", history.history['val_root_mean_squared_error'][-1])
    mlflow.log_metric("epochs_trained", len(history.history['loss']))
    
    # Create and log training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['root_mean_squared_error'], label='Training RMSE')
    ax2.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
    ax2.set_title('Model RMSE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    mlflow.log_figure(fig, "training_curves.png")
    plt.close()

def log_tensorflow_model_with_artifacts(model, X_test, model_name="BMRI_LSTM_Model"):
    """Log TensorFlow model with signature and artifacts"""
    # Log model with signature
    sample_input = X_test[:5]
    sample_prediction = model.predict(sample_input, verbose=0)
    signature = infer_signature(sample_input, sample_prediction)
    
    mlflow.tensorflow.log_model(
        model, 
        "model", 
        signature=signature,
        registered_model_name=model_name
    )
    
    # Save and log H5 model file
    h5_filename = f"{model_name.lower()}.h5"
    model.save(h5_filename)
    mlflow.log_artifact(h5_filename)
    os.remove(h5_filename)

def register_best_model(rmse, model_name="BMRI_Best_Model", experiment_name="BMRI_Stock_Prediction"):
    """Register model as best if it has lowest RMSE"""
    try:
        client = mlflow.tracking.MlflowClient()
        current_run = mlflow.active_run()
        
        # Check if this is the best model
        experiment = client.get_experiment_by_name(experiment_name)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["metrics.test_rmse ASC"]
        )
        
        is_best_model = True
        if len(runs) > 1:
            best_previous_rmse = runs[1].data.metrics.get('test_rmse', float('inf'))
            is_best_model = rmse < best_previous_rmse
        
        if is_best_model:
            try:
                client.create_registered_model(model_name)
            except:
                pass
            
            model_uri = f"runs:/{current_run.info.run_id}/model"
            model_version = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=current_run.info.run_id
            )
            
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production"
            )
            
            mlflow.log_param("registered_as_best_model", True)
            mlflow.log_param("model_version", model_version.version)
            print(f"🏆 Model registered as best model - Version {model_version.version}")
        else:
            mlflow.log_param("registered_as_best_model", False)
            print(f"📊 Model performance: RMSE {rmse:.2f} (not best)")
            
    except Exception as e:
        print(f"⚠️ Error in model registration: {e}")
        mlflow.log_param("registration_error", str(e))

def end_mlflow_run():
    """End the current MLflow run"""
    mlflow.end_run()

def print_mlflow_ui_info():
    """Print MLflow UI access information"""
    print(f"\n🔍 View experiment results at: http://localhost:5000")
    print(f"   Run: mlflow ui")