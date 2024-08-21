# Amazon Product Project

## Description

This project processes raw Amazon product data, cleans it, trains a machine learning model to predict product prices, and saves the trained model.

## Directory Structure

- `data/raw/`: Contains the raw CSV data.
- `data/processed/`: Contains the cleaned CSV data.
- `models/`: Contains the trained model.
- `logs/`: Contains log files.
- `notebooks/`: Contains Jupyter notebooks for exploration.
- `src/`: Contains source code for data processing, modeling, and the main script.
- `tests/`: Contains test scripts.

## Requirements

- `pandas`
- `scikit-learn`
- `joblib`
- `python-dotenv`
- `scipy`
- `xgboost`

## Setup

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Create a `.env` file with appropriate paths for data, models, and logs.

## Running the Project

To run the data processing and model training pipeline:

```bash
python src/main.py
