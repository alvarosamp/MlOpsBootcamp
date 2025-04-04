import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from MLOpsBootcamp.PackagingMLModels.packaging_ml_model.config.config import config
from MLOpsBootcamp.PackagingMLModels.packaging_ml_model.predict import generate_predictions
from MLOpsBootcamp.PackagingMLModels.packaging_ml_model.processing.data_handling import load_dataset, separate_data, split_data, save_pipeline, load_pipeline

#Output from predict not null
#output from predict script is str data type
#The output is Y for an example data 

@pytest.fixture
def single_prediction():
    test_data = load_dataset(config.TEST_FILE)
    single_row = test_data[:1]
    result = generate_predictions(single_row)
    return result

def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None, "Prediction result should not be None"

def test_single_pred_type(single_prediction):
    assert isinstance(single_prediction, str), "Prediction result should be of type str"

def test_single_pred_validate(single_prediction):
    # Assuming the expected prediction for the test data is 'Y'
    expected_prediction = 'Y'
    assert single_prediction == expected_prediction, f"Expected prediction: {expected_prediction}, but got: {single_prediction}"