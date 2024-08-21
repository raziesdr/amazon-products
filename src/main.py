import logging
from config import RAW_DATA_PATH, MODEL_PATH, LOG_PATH, PROCESSED_DATA_PATH
from src.preprocessing.data_processor import DataProcessor
from src.modeling.price_predictor import PricePredictor

# Configure logging
logging.basicConfig(filename=LOG_PATH,
                    filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


def main():
    """Main function to run the entire pipeline: data processing, model training, and prediction."""
    try:
        # Load and process the data
        processor = DataProcessor(file_path=RAW_DATA_PATH)
        data = processor.load_and_clean_data()

        if data is None:
            logging.error("Data loading or cleaning failed.")
            return

        # Feature engineering
        processed_data = processor.feature_engineering(data)

        if processed_data is None:
            logging.error("Feature engineering failed.")
            return

        # Save the preprocessed data to a CSV file
        processor.save_processed_data(processed_data, PROCESSED_DATA_PATH)

        # Initialize and train the model
        predictor = PricePredictor(data=processed_data)
        predictor.train_models()

        # Make future predictions
        predictions = predictor.predict(future_steps=10)
        logging.info(f"Predictions for the next 10 days: \n{predictions}")

        # Save the trained model
        predictor.save_model(MODEL_PATH)

    except Exception as e:
        logging.error(f"Error in main pipeline: {e}")


if __name__ == "__main__":
    main()
