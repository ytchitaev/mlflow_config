import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

class DatasetSplitter:
    
    def split_dataset(logger, X_input: pd.DataFrame, y_input: pd.Series, train_percentage: float, validation_percentage: float,
                    test_percentage: float, random_state: int=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
                                                    pd.Series]:
        logger.info("Splitting data...")                                        
        X_train, X_remaining, y_train, y_remaining = train_test_split(
            X_input, y_input, test_size=(validation_percentage + test_percentage) / 100, random_state=random_state)
        X_validation, X_test, y_validation, y_test = train_test_split(
            X_remaining, y_remaining, test_size=test_percentage / (validation_percentage + test_percentage),
            random_state=random_state)
        return X_train, X_validation, X_test, y_train, y_validation, y_test