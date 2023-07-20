import unittest
from unittest.mock import MagicMock
import pandas as pd
import pyodbc
from pandas._testing import assert_frame_equal, assert_series_equal
from functions.data_loader import CSVDataLoader, AzureSQLDataLoader


class TestDataLoader(unittest.TestCase):

    def test_csv_data_loader(self):
        data_loader = CSVDataLoader()
        input_columns = ['column1', 'column2']
        output_columns = ['target']
        csv_file_path = 'path/to/csv/file.csv'

        # Create a sample DataFrame for testing
        data = {
            'column1': [1, 2, 3],
            'column2': [4, 5, 6],
            'target': [0, 1, 0]
        }
        df = pd.DataFrame(data)

        # Mock the pd.read_csv function to return the sample DataFrame
        pd.read_csv = MagicMock(return_value=df)

        X, y = data_loader.load_data(
            input_columns, output_columns, csv_file_path)

        assert_frame_equal(X, df[input_columns])
        assert_series_equal(y, df[output_columns])

    def test_azure_sql_data_loader(self):
        data_loader = AzureSQLDataLoader()
        input_columns = ['column1', 'column2']
        output_columns = ['target']
        connection_string = 'connection_string'
        table_name = 'table_name'

        # Create a sample DataFrame for testing
        data = {
            'column1': [1, 2, 3],
            'column2': [4, 5, 6],
            'target': [0, 1, 0]
        }
        df = pd.DataFrame(data)

        # Mock the pyodbc.connect and pd.read_sql functions to return the sample DataFrame
        pyodbc.connect = MagicMock()
        pd.read_sql = MagicMock(return_value=df)

        X, y = data_loader.load_data(
            input_columns, output_columns, connection_string, table_name)

        assert_frame_equal(X, df[input_columns])
        assert_series_equal(y, df[output_columns])


if __name__ == '__main__':
    unittest.main()
