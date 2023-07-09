import unittest
from unittest.mock import patch, MagicMock
from pandas.testing import assert_frame_equal
import pandas as pd
from functions.data_loader import SklearnDataLoader, CSVDataLoader, AzureSQLDataLoader, load_data

class TestDataLoaders(unittest.TestCase):
    @patch('functions.data_loader.datasets')
    def test_sklearn_data_loader(self, mock_datasets):
        mock_data = MagicMock()
        mock_data.data = pd.DataFrame({'feature': [1, 2, 3]})
        mock_data.target = pd.Series([0, 1, 0])
        mock_datasets.load_iris.return_value = mock_data

        loader = SklearnDataLoader()
        X, y = loader.load_data('iris')

        assert_frame_equal(X, mock_data.data)
        self.assertTrue(y.equals(mock_data.target))

    @patch('functions.data_loader.pd.read_csv')
    def test_csv_data_loader(self, mock_read_csv):
        mock_df = pd.DataFrame({'feature': [1, 2, 3], 'target': [0, 1, 0]})
        mock_read_csv.return_value = mock_df

        loader = CSVDataLoader()
        X, y = loader.load_data('path/to/file.csv')

        assert_frame_equal(X, mock_df.iloc[:, :-1])
        self.assertTrue(y.equals(mock_df.iloc[:, -1]))

    @patch('functions.data_loader.pyodbc')
    @patch('functions.data_loader.pd.read_sql')
    def test_azure_sql_data_loader(self, mock_read_sql, mock_pyodbc):
        mock_df = pd.DataFrame({'feature': [1, 2, 3], 'target': [0, 1, 0]})
        mock_read_sql.return_value = mock_df

        loader = AzureSQLDataLoader()
        X, y = loader.load_data('connection_string', 'table_name')

        assert_frame_equal(X, mock_df.iloc[:, :-1])
        self.assertTrue(y.equals(mock_df.iloc[:, -1]))

    def test_load_data(self):
        with self.assertRaises(ValueError):
            load_data('unsupported_source')
        
        loader_classes = {
            'sklearn': SklearnDataLoader,
            'csv': CSVDataLoader,
            'azure_sql': AzureSQLDataLoader
        }

        for source in loader_classes.keys():
            with patch.object(loader_classes[source], 'load_data') as mock_load_data:
                load_data(source)
                mock_load_data.assert_called_once()

if __name__ == '__main__':
    unittest.main()
