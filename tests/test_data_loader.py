import unittest
from unittest.mock import MagicMock
import pandas as pd
from functions.data_loader import load_data, SklearnDataLoader, CSVDataLoader, AzureSQLDataLoader
from sklearn import datasets
import pyodbc


class TestDataLoader(unittest.TestCase):
    def test_load_data_with_sklearn_data_loader(self):
        data_source = 'sklearn'
        input_columns = ['column1', 'column2']
        output_columns = ['target']
        dataset_name = 'iris'

        # Mock the datasets.load_iris function
        datasets.load_iris = MagicMock(return_value=pd.DataFrame({'data': [[1, 2], [3, 4]], 'target': [0, 1], 'feature_names': ['column1', 'column2']}))

        # Test the load_data function
        X, y = load_data(None, data_source, input_columns, output_columns, dataset_name=dataset_name)
        datasets.load_iris.assert_called_once()
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(y.shape, (2,))

    def test_load_data_with_csv_data_loader(self):
        data_source = 'csv'
        input_columns = ['column1', 'column2']
        output_columns = ['target']
        csv_file_path = 'path/to/csv'

        # Mock the pd.read_csv function
        pd.read_csv = MagicMock(return_value=pd.DataFrame({'column1': [1, 2], 'column2': [3, 4], 'target': [0, 1]}))

        # Test the load_data function
        X, y = load_data(None, data_source, input_columns, output_columns, csv_file_path=csv_file_path)
        pd.read_csv.assert_called_once_with(csv_file_path, delimiter=',', encoding='utf-8')
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(y.shape, (2,))

    def test_load_data_with_azure_sql_data_loader(self):
        data_source = 'azure_sql'
        input_columns = ['column1', 'column2']
        output_columns = ['target']
        connection_string = 'connection_string'
        table_name = 'table_name'

        # Mock the pyodbc.connect and pd.read_sql functions
        pyodbc.connect = MagicMock()
        pd.read_sql = MagicMock(return_value=pd.DataFrame({'column1': [1, 2], 'column2': [3, 4], 'target': [0, 1]}))

        # Test the load_data function
        X, y = load_data(None, data_source, input_columns, output_columns, connection_string=connection_string, table_name=table_name)
        pyodbc.connect.assert_called_once_with(connection_string)
        pd.read_sql.assert_called_once_with(f"SELECT * FROM {table_name}", pyodbc.connect.return_value)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(y.shape, (2,))

    # Add more test cases for other data loader classes...


if __name__ == '__main__':
    unittest.main()
