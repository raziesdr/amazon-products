import pandas as pd
import xgboost as xgb
import pickle
import logging
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class PricePredictor:
    """Predicts the next prices and quantities using XGBoost models."""

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the PricePredictor with pivoted data.
        Args:
            data (pd.DataFrame): The pivoted data for training.
        """
        self.data = data
        self.models = {}

    def _prepare_data_for_modeling(self, column: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepares data for modeling by creating lag features.
        Args:
            column (str): The target column for which to predict.
        Returns:
            Optional[Tuple[pd.DataFrame, pd.Series]]: Features and target for modeling, or None if insufficient data.
        """
        df = self.data[[column]].copy()
        df.fillna(method='ffill', inplace=True)

        df['lag_1'] = df[column].shift(1)
        df['lag_2'] = df[column].shift(2)
        df['lag_3'] = df[column].shift(3)
        df.dropna(inplace=True)

        if df.shape[0] == 0:
            logging.warning(f"Insufficient data after creating lag features for column: {column}")
            return None

        X = df[['lag_1', 'lag_2', 'lag_3']]
        y = df[column]
        return X, y

    def train_models(self) -> None:
        """Trains XGBoost models for each product's price and quantity."""
        try:
            for column in self.data.columns:
                if self.data[column].dtype == 'object' or column == 'date':
                    logging.info(f"Skipping non-numerical column: {column}")
                    continue

                result = self._prepare_data_for_modeling(column)
                if result is None:
                    logging.error(f"Skipping model training for {column} due to insufficient data.")
                    continue

                X, y = result
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
                model.fit(X_train, y_train)
                self.models[column] = model

                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                logging.info(f"Model for {column} trained. MSE: {mse}")

        except Exception as e:
            logging.error(f"Error in model training: {e}")

    def predict(self, future_steps: int = 7) -> pd.DataFrame:
        """
        Predicts future values for each product's price and quantity.
        Args:
            future_steps (int): Number of future steps to predict.
        Returns:
            pd.DataFrame: DataFrame containing future predictions.
        """
        try:
            future_predictions = {}
            for column, model in self.models.items():
                last_known_values = self.data[[column]].iloc[-3:].values.flatten().tolist()
                predictions = []
                for _ in range(future_steps):
                    X_new = pd.DataFrame([last_known_values[-3:]], columns=['lag_1', 'lag_2', 'lag_3'])
                    y_pred = model.predict(X_new)[0]
                    predictions.append(y_pred)
                    last_known_values.append(y_pred)
                future_predictions[column] = predictions
            future_df = pd.DataFrame(future_predictions)
            logging.info("Future predictions completed successfully.")
            return future_df

        except Exception as e:
            logging.error(f"Error in making predictions: {e}")
            return pd.DataFrame()

    def save_model(self, model_path: str) -> None:
        """
        Saves the trained models to a pickle file.
        Args:
            model_path (str): The path to save the model.
        """
        try:
            with open(model_path, 'wb') as model_file:
                pickle.dump(self.models, model_file)
            logging.info(f"Models saved to {model_path}")

        except Exception as e:
            logging.error(f"Error saving models: {e}")


