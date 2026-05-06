# Sales Forecasting System

Production-ready Python/FastAPI project for state-level sales forecasting from an Excel dataset.

For a company-review friendly overview of the project, see [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md).

## Features

- Loads sales data from Excel.
- Handles missing dates and missing sales values per state.
- Builds lag, rolling, calendar, and holiday features.
- Trains and compares SARIMA, Prophet, XGBoost, and TensorFlow/Keras LSTM.
- Uses time-series train/validation split.
- Evaluates RMSE and MAE.
- Selects and persists the best model for each state.
- Tracks runs and metrics in MLflow.
- Saves forecast visualization plots for each state.
- Saves model comparison reports and ranked best-model summaries.
- Serves training, prediction, and model registry endpoints with FastAPI.

## Project Structure

```text
data/                 Excel input files
models/               Saved model artifacts and registry
artifacts/            Model comparison reports
artifacts/plots/      Forecast visualization PNG files
logs/                 Rotating application logs
api/                  FastAPI application and schemas
training/             Training pipeline, features, metrics, model wrappers
training/models/      SARIMA, Prophet, XGBoost, and LSTM implementations
utils/                Configuration, logging, exceptions, model IO
```

## Expected Dataset

By default, place your Excel file at:

```text
data/sales.xlsx
```

The pipeline automatically detects common column names for:

```text
date, state, sales
```

Examples that are detected include `Order Date`, `Week Start`, `State`, `Region`, `Sales`, `Revenue`, `Amount`, and `holiday_flag`. Override column names with environment variables if your uploaded file uses unusual names:

```powershell
$env:DATE_COLUMN="Order Date"
$env:STATE_COLUMN="State"
$env:SALES_COLUMN="Sales"
```

Rows with duplicate `state` and `date` values are summed. Missing daily dates are inserted per state, and missing sales values are filled with time interpolation plus forward/backward fallback.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Prophet and TensorFlow can take longer to install and may require platform-specific wheels. Use Python 3.10 or 3.11 for the smoothest compatibility.

## Train From CLI

Train all models for every state:

```powershell
python -m training.train --excel-path data/sales.xlsx
```

Train only selected models:

```powershell
python -m training.train --excel-path data/sales.xlsx --models sarima xgboost
```

Model artifacts are written under `models/`, forecast plots are written under `artifacts/plots/`, model comparison reports are written to `artifacts/model_comparison.csv` and `artifacts/model_comparison.json`, and MLflow runs are written under `mlruns/` by default.

## Run API

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

## API Endpoints

### POST `/train`

```json
{
  "excel_path": "data/sales.xlsx"
}
```

The endpoint trains SARIMA, Prophet, XGBoost, and LSTM for all detected states and returns validation metrics.

### POST `/predict`

Forecast one state:

```json
{
  "state": "California",
  "weeks_ahead": 8
}
```

The response includes weekly forecast values and the selected best model.

### GET `/models`

Returns available trained models, selected best model, and validation metrics per state.

## Configuration

Key environment variables:

```text
SALES_EXCEL_PATH=data/sales.xlsx
DATA_DIR=data
MODEL_DIR=models
LOGS_DIR=logs
LOG_LEVEL=INFO
LOG_MAX_BYTES=5242880
LOG_BACKUP_COUNT=5
MLFLOW_TRACKING_URI=file:///absolute/path/to/mlruns
MLFLOW_EXPERIMENT_NAME=retail_sales_forecasting
FORECAST_HORIZON_DAYS=56
VALIDATION_DAYS=56
MINIMUM_TRAINING_DAYS=90
LSTM_SEQUENCE_LENGTH=30
```

## MLflow

Training writes all model runs to the `retail_sales_forecasting` experiment. Each SARIMA, Prophet, XGBoost, and LSTM run logs:

```text
model_type, state, RMSE, MAE, training_time, hyperparameters, metrics, parameters, artifacts
```

Start the MLflow UI:

```powershell
mlflow ui --backend-store-uri .\mlruns
```

Then open:

```text
http://localhost:5000
```

## Logging

Application logs are written to `logs/app.log` with rotating file backups. Logs include timestamps, log level, API endpoint activity, training events, prediction requests, model selection, and errors with stack traces.

Rotation is controlled by:

```text
LOG_MAX_BYTES=5242880
LOG_BACKUP_COUNT=5
```

## Notes

- The split is chronological: the last `VALIDATION_DAYS` are reserved for validation.
- XGBoost forecasts recursively using the required lag and rolling features.
- SARIMA uses a weekly seasonal component by default.
- Prophet uses weekly and yearly seasonality.
- LSTM uses the configured sequence length and saves both the Keras model and scaler.
- Best model selection is based on lowest validation RMSE per state.
