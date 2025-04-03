from sklearn.pipeline import Pipeline
"""
This module defines a machine learning pipeline for a classification task using scikit-learn's `Pipeline` class.
Pipeline Steps:
1. **DomainProcessing**:
    - Applies domain-specific processing to the data.
    - Adds a new feature to the dataset based on the variable specified in `config.FEATURE_TO_ADD`.
2. **DropFeatures**:
    - Drops unnecessary or irrelevant features from the dataset.
    - The features to drop are specified in `config.DROP_FEATURES`.
3. **LabelEncoder**:
    - Encodes categorical variables into numerical format.
    - The variables to encode are specified in `config.FEATURES_TO_ENCODE`.
4. **LogTransform**:
    - Applies logarithmic transformation to specified features.
    - The features to transform are specified in `config.LOG_FEATURES`.
5. **LogisticClassifier**:
    - A logistic regression model is used as the final estimator for classification.
    - The model is initialized with a fixed random state for reproducibility.
"""
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
from packaging_ml_model.config import config
import packaging_ml_model.processing.preprocessing as pp
from sklearn.linear_model import LogisticRegression
import numpy as np

classification_pipeline = Pipeline(
    [
        ('DomainProcessing', pp.DomainProcessing(variable_to_add=config.FEATURE_TO_ADD)),
        ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('LabelEncoder', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('LogTransform', pp.LogTransforms(variables=config.LOG_FEATURES)),
        ('LogisticClassifier', LogisticRegression(random_state=0))
    ]
)
