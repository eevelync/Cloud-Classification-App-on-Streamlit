import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from app import process_data_and_predict

class MockModel:
    def predict(self, df):
        return [0] * len(df)

    def predict_proba(self, df):
        return [[0.5, 0.5]] * len(df)

class TestProcessDataAndPredictions:
    mock_model = MockModel()
    features = ['feature1', 'feature2']
    df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

    @pytest.mark.parametrize('bad_features', [['non_existent_feature']])
    @patch('logging.error')
    @patch('streamlit.error')
    def test_unhappy_path(self, mock_st_error, mock_logging_error, bad_features):
        predictions_df, probabilities_df = process_data_and_predict(self.mock_model, self.df, bad_features)

        assert predictions_df.empty
        assert probabilities_df.empty

        mock_logging_error.assert_called_once()
        mock_st_error.assert_called_once_with('Unexpected error during prediction. Please check the logs for more details.')

    @pytest.mark.parametrize('expected_predictions_df, expected_probabilities_df', [
        (pd.DataFrame([0, 0, 0], columns=['prediction']), pd.DataFrame([[0.5, 0.5]] * 3, columns=['prob class1', 'prob class2']).applymap(lambda x: f'{x:.2f}'))
    ])
    def test_happy_path(self, expected_predictions_df, expected_probabilities_df):
        predictions_df, probabilities_df = process_data_and_predict(self.mock_model, self.df, self.features)

        pd.testing.assert_frame_equal(predictions_df, expected_predictions_df)
        pd.testing.assert_frame_equal(probabilities_df, expected_probabilities_df)
