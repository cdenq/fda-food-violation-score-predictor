#----------------------------------------------------
# Libraries
#----------------------------------------------------
# EDA / Visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Computation
import math
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Formatting / Outputting
from IPython.display import Image
import dataframe_image as dfi

# Modeling (Logistic Regression)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Modeling (Sentiment Analysis)
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Deployment
import joblib
import streamlit as st

#----------------------------------------------------
# Global Variables
#----------------------------------------------------
RANDOM_SEED = 42
SERIOUS_VIOLATION_SCORE_THRESHOLD = 10
CHARS_TO_REMOVE = ["[", "]", "'", " ", "/", "-", "(", ")", "&"]

#----------------------------------------------------
# FILEPATH Variables
#----------------------------------------------------
DEV_PATH_TO_EDA = "../outputs/eda_outputs"
DEV_PATH_TO_MODEL = "../outputs/model_outputs"
DEV_PATH_TO_SAVED_MODELS = "../outputs/saved_models"
DEV_PATH_TO_RAW_DATA = "../data/raw"
DEV_PATH_TO_PREPPED_DATA = "../data/prepped"

APP_PATH_TO_SAVED_MODELS = "outputs/saved_models"
APP_PATH_TO_RAW_DATA = "data/raw"
APP_PATH_TO_PREPPED_DATA = "data/prepped"

#----------------------------------------------------
# Default Variables
#----------------------------------------------------
DEFAULT_GRID_ALPHA = 0.5
DEFAULT_GRAPH_ALPHA = 0.7
DEFAULT_MARKER_SIZE = 8
DEFAULT_BAR_WIDTH = 0.25
DEFAULT_LONG_FIG_SIZE = (6, 4)
DEFAULT_TALL_FIG_SIZE = (4, 6)
DEFAULT_SQUARE_FIG_SIZE = (4, 4)
DEFAULT_BIG_FIG_SIZE = (8, 6)

DEFAULT_TRAIN = 0.75
DEFAULT_VAL = 0.10
DEFAULT_TEST = 0.15
DEFAULT_FOLDS = 5

#----------------------------------------------------
# Main
#----------------------------------------------------
def main():
    return None

#----------------------------------------------------
# Entry
#----------------------------------------------------
if __name__ == "__main__":
    main()