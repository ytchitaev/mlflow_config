import pandas as pd
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any, Tuple


def perform_grid_search(model: Any, X_train: pd.DataFrame, y_train: pd.Series, param_grid: Dict[str, Any], cv: int) -> GridSearchCV:
    grid_search = GridSearchCV(model, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.cv_results_
