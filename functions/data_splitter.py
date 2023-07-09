import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

def split_dataset(X_input: pd.DataFrame, y_input: pd.Series, train_percentage: float, validation_percentage: float,
                  test_percentage: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
                                                   pd.Series]:
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X_input, y_input, test_size=(validation_percentage + test_percentage) / 100, random_state=42)
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining, y_remaining, test_size=test_percentage / (validation_percentage + test_percentage),
        random_state=42)
    return X_train, X_validation, X_test, y_train, y_validation, y_test