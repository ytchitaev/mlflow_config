from abc import ABC, abstractmethod
from typing import Tuple, Any
import pandas as pd
import pyodbc
from sklearn import datasets
#from snowflake import connector
#from pyspark.sql import SparkSession


class DataLoader(ABC):
    @abstractmethod
    def load_data(self, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        pass


class SklearnDataLoader(DataLoader):
    def load_data(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        load_func = getattr(datasets, f"load_{dataset_name}")
        data = load_func()
        X, y = data.data, data.target
        return X, y


class CSVDataLoader(DataLoader):
    def load_data(self, csv_file_path: str, delimiter: str = ',', encoding: str = 'utf-8') -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(csv_file_path, delimiter=delimiter, encoding=encoding)
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        return X, y


class AzureSQLDataLoader(DataLoader):
    def load_data(self, connection_string: str, table_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        conn = pyodbc.connect(connection_string)
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        return X, y


# not yet implemented

#class DatabricksDeltaLakeDataLoader(DataLoader):
#    def load_data(self, table_name: str) -> Tuple[pd.DataFrame, pd.Series]:
#        spark = SparkSession.builder.getOrCreate()
#        df = spark.sql(f"SELECT * FROM delta.`{table_name}`")
#        X, y = df.drop('target'), df.select('target')
#        return X, y


#class SnowflakeDataLoader(DataLoader):
#    def load_data(self, connection_string: str, table_name: str) -> Tuple[pd.DataFrame, pd.Series]:
#        conn = connector.connect(connection_string)
#        query = f"SELECT * FROM {table_name}"
#        df = pd.read_sql(query, conn)
#        X, y = df.iloc[:, :-1], df.iloc[:, -1]
#        return X, y


def load_data(data_source: str, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
    loader_classes = {
        'sklearn': SklearnDataLoader,
        'csv': CSVDataLoader,
        'azure_sql': AzureSQLDataLoader #,
        #'databricks_delta_lake': DatabricksDeltaLakeDataLoader,
        #'snowflake': SnowflakeDataLoader,
    }

    if data_source not in loader_classes:
        raise ValueError(f"Unsupported data source: {data_source}")

    loader = loader_classes[data_source]()
    return loader.load_data(**kwargs)
