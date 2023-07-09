import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV


def instantiate_model(libary_name, model_name, params={}):
    if libary_name == 'lightgbm':
        if model_name == 'LGBMClassifier':
            model = lgb.LGBMClassifier(**params)
        elif model_name == 'LGBMRegressor':
            model = lgb.LGBMRegressor(**params)
        elif model_name == 'LGBMRanker':
            model = lgb.LGBMRanker(**params)
    else:
        raise ValueError(f"Unsupported libary_name: {libary_name} - model name: {model_name}")
    return model


def split_dataset(X_input, y_input, train_percentage, validation_percentage, test_percentage):
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X_input, y_input, test_size=(validation_percentage + test_percentage) / 100, random_state=42)
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining, y_remaining, test_size=test_percentage / (validation_percentage + test_percentage), random_state=42)
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def perform_grid_search(model, X_train, y_train, param_grid, cv):
    grid_search = GridSearchCV(model, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    return grid_search
