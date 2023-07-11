import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from functions.data_loader import SklearnDataLoader, CSVDataLoader, AzureSQLDataLoader, load_data

class TestDataLoaders(unittest.TestCase):

    def test_sklearn_data_loader(self):
        with patch('functions.data_loader.datasets') as mock_datasets:
            mock_data = MagicMock()
            mock_data.data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
            mock_data.target = pd.Series([0, 1, 0])
            mock_data.feature_names = ['feature1', 'feature2']
            mock_datasets.load_iris.return_value = mock_data

            loader = SklearnDataLoader()
            X, y = loader.load_data(['feature1'], ['target'], 'iris')

            pd.testing.assert_frame_equal(X, mock_data.data[['feature1']])
            self.assertTrue(y.equals(mock_data.target))

    def test_csv_data_loader(self):
        with patch('functions.data_loader.pd.read_csv') as mock_read_csv:
            mock_df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'target': [0, 1, 0]})
            mock_read_csv.return_value = mock_df

            loader = CSVDataLoader()
            X, y = loader.load_data(['feature1'], ['target'], 'path/to/file.csv')

            pd.testing.assert_frame_equal(X, mock_df[['feature1']])
            self.assertTrue(y.equals(mock_df[['target']]))

    def test_azure_sql_data_loader(self):
        with patch('functions.data_loader.pyodbc.connect'), \
            patch('functions.data_loader.pd.read_sql') as mock_read_sql:

            mock_df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'target': [0, 1, 0]})
            mock_read_sql.return_value = mock_df

            loader = AzureSQLDataLoader()
            X, y = loader.load_data(['feature1'], ['target'], 'connection_string', 'table_name')

            pd.testing.assert_frame_equal(X, mock_df[['feature1']])
            self.assertTrue(y.equals(mock_df[['target']]))


    def test_load_data_unsupported(self):
        with self.assertRaises(ValueError) as context:
            load_data('unsupported_source', ['feature'], ['target'])
        self.assertTrue('Unsupported data source: unsupported_source' in str(context.exception))

    def test_load_data_supported(self):
        loader_classes = {
            'sklearn': SklearnDataLoader,
            'csv': CSVDataLoader,
            'azure_sql': AzureSQLDataLoader
        }

        for source, loader_class in loader_classes.items():
            with patch.object(loader_class, 'load_data') as mock_load_data:
                load_data(source, ['feature'], ['target'])
                mock_load_data.assert_called_once_with(['feature'], ['target'])


if __name__ == '__main__':
    unittest.main()
