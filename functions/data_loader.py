from sklearn import datasets
import pandas as pd
import pyodbc
#from snowflake import connector
# from pyspark.sql import SparkSession


def load_data(source, **kwargs):
    if source == 'sklearn':
        return load_sklearn_dataset(kwargs['dataset_name'])
    elif source == 'csv':
        return load_from_csv(kwargs['csv_file_path'])
    elif source == 'azure_sql':
        return load_from_azure_sql(kwargs['connection_string'], kwargs['table_name'])
    # elif source == 'databricks_delta_lake':
    #    return load_from_databricks_delta_lake(kwargs['table_name'])
    #elif source == 'snowflake':
    #    return load_from_snowflake(kwargs['connection_string'], kwargs['table_name'])
    else:
        raise ValueError(f"Unsupported data source: {source}")


def load_sklearn_dataset(dataset_name):
    load_func = getattr(datasets, f"load_{dataset_name}")
    data = load_func()
    X, y = data.data, data.target
    return X, y


def load_from_csv(csv_file_path, delimiter=',', encoding='utf-8'):
    df = pd.read_csv(csv_file_path, delimiter=delimiter, encoding=encoding)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    return X, y


def load_from_azure_sql(connection_string, table_name):
    conn = pyodbc.connect(connection_string)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    return X, y


# def load_from_databricks_delta_lake(table_name):
#    spark = SparkSession.builder.getOrCreate()
#    df = spark.sql(f"SELECT * FROM delta.`{table_name}`")
#    X, y = df.drop('target'), df.select('target')
#    return X, y


#def load_from_snowflake(connection_string, table_name):
#    conn = connector.connect(connection_string)
#    query = f"SELECT * FROM {table_name}"
#    df = pd.read_sql(query, conn)
#    X, y = df.iloc[:, :-1], df.iloc[:, -1]
#    return X, y