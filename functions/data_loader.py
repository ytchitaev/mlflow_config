from abc import ABC, abstractmethod
from typing import Tuple, Any, List
import pandas as pd
import pyodbc
from sklearn import datasets
#from snowflake import connector
#from pyspark.sql import SparkSession


class DataLoader(ABC):
    def __init__(self):
        self.input_columns = None,
        self.output_columns = None
        
    @abstractmethod
    def load_data(self, column_names: Tuple[List[str], List[str]], **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        pass

class SklearnDataLoader(DataLoader):
    def load_data(self, column_names: Tuple[List[str], List[str]], dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        load_func = getattr(datasets, f"load_{dataset_name}")
        data = load_func()
        # output column filtering is not enabled as sklearn does not always make the output column name available
        # output column name is driven by config
        self.input_columns = [col for col in data.feature_names if col in column_names[0]]
        X = pd.DataFrame(data.data, columns=data.feature_names)[self.input_columns]
        y = pd.Series(data.target)
        return X, y


class CSVDataLoader(DataLoader):
    def load_data(self, column_names: Tuple[List[str], List[str]], csv_file_path: str, delimiter: str = ',', encoding: str = 'utf-8') -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(csv_file_path, delimiter=delimiter, encoding=encoding)
        self.input_columns, self.output_columns = column_names
        X = df[self.input_columns]
        y = df[self.output_columns]
        return X, y


class AzureSQLDataLoader(DataLoader):
    def load_data(self, column_names: Tuple[List[str], List[str]], connection_string: str, table_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        conn = pyodbc.connect(connection_string)
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)
        self.input_columns, self.output_columns = column_names
        X = df[self.input_columns]
        y = df[self.output_columns]
        return X, y


#class DatabricksDeltaLakeDataLoader(DataLoader):
#    def load_data(self, column_names: Tuple[List[str], List[str]], table_name: str) -> Tuple[pd.DataFrame, pd.Series]:
#        spark = SparkSession.builder.getOrCreate()
#        df = spark.sql(f"SELECT * FROM delta.`{table_name}`")
#        self.input_columns, self.output_columns = column_names
#        X = df.select(self.input_columns)
#        y = df.select(self.output_columns)
#        return X, y


#class SnowflakeDataLoader(DataLoader):
#    def load_data(self, column_names: Tuple[List[str], List[str]], connection_string: str, table_name: str) -> Tuple[pd.DataFrame, pd.Series]:
#        conn = connector.connect(connection_string)
#        query = f"SELECT * FROM {table_name}"
#        df = pd.read_sql(query, conn)
#        self.input_columns, self.output_columns = column_names
#        X = df[self.input_columns]
#        y = df[self.output_columns]
#        return X, y


def load_data(data_source: str, column_names: Tuple[List[str], List[str]], **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
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
    data = loader.load_data(column_names, **kwargs)
    return data
