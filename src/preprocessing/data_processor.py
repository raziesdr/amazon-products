import pandas as pd
import logging
from src.config import LOG_PATH


class DataProcessor:
    """Processes and cleans the raw data."""

    def __init__(self, file_path):
        """
        Initializes the DataProcessor with the path to the raw data.
        Args:
            file_path (str): Path to the CSV file.
        """
        self.file_path = file_path
        logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def load_and_clean_data(self):
        """
        Loads data from a CSV file, cleans it, and returns a DataFrame.

        Returns:
            pd.DataFrame: Cleaned data.
        """
        try:
            df = pd.read_csv(self.file_path)
            logging.info("Data loaded successfully.")
            logging.info(f"Columns in the dataset: {df.columns}")
            logging.info(f"First few rows:\n{df.head()}")

            # Drop unwanted columns
            df = df.drop(columns=['Unnamed: 0'], errors='ignore')

            # Log initial data size
            logging.info(f"Data size before cleaning: {df.shape}")

            # Log raw values in 'actual_price'
            logging.info(f"Raw values in 'actual_price':\n{df['actual_price'].head()}")

            # Clean 'actual_price' column
            df['actual_price'] = pd.to_numeric(df['actual_price'].str.replace(',', '').str.replace('₹', ''),
                                               errors='coerce')
            logging.info("'actual_price' column cleaned and converted to numeric.")

            # Log cleaned values in 'actual_price'
            logging.info(f"Cleaned values in 'actual_price':\n{df['actual_price'].head()}")

            # Handle missing values in 'actual_price'
            missing_values = df['actual_price'].isnull().sum()
            logging.info(f"Missing values in 'actual_price': {missing_values}")

            df = df.dropna(subset=['actual_price'])
            logging.info(f"Data size after dropping missing values: {df.shape}")

            # Convert 'date' column to datetime
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            logging.info("'date' column converted to datetime.")

            # Drop rows where 'date' is NaT
            df = df.dropna(subset=['date'])
            logging.info(f"Data size after dropping missing dates: {df.shape}")

            logging.info(f"Data after cleaning:\n{df.head()}")
            return df

        except KeyError as e:
            logging.error(f"Error in data loading/cleaning: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return None

    def feature_engineering(self, df):
        """
        Creates a pivot table from the data for feature engineering.

        Args:
            df (pd.DataFrame): DataFrame with cleaned data.

        Returns:
            pd.DataFrame: Pivoted DataFrame.
        """
        try:
            if df.empty:
                logging.error("DataFrame is empty. Cannot perform feature engineering.")
                return None

            # Pivot table to get the total number of products and average price per product per day
            pivot_df = df.pivot_table(
                index='date',
                columns='name',
                values=['actual_price', 'no_of_ratings'],
                aggfunc={'actual_price': 'mean', 'no_of_ratings': 'sum'}
            )
            pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
            pivot_df.reset_index(inplace=True)

            # Check for an empty pivot table
            if pivot_df.empty:
                logging.warning(
                    "Pivot table is empty. Ensure that the 'name' column and values are correctly specified.")

            logging.info("Feature engineering completed successfully.")
            logging.info(f"Pivoted DataFrame:\n{pivot_df.head()}")
            return pivot_df

        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            return None

    def save_processed_data(self, df, path):
        """
        Save the processed DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): DataFrame to save.
            path (str): Path to the CSV file.
        """
        try:
            df.to_csv(path, index=False)
            logging.info(f"Processed data saved to {path}")
        except Exception as e:
            logging.error(f"Error saving processed data: {e}")
