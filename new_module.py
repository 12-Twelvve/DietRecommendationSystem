import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')


def calculate_bmi(height, weight):
    height /= 100
    bmi = weight/np.square(height)
    return round(bmi,2)

def get_min_max(col_name, df):
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3-q1
    q_min = q1-(1.5*iqr)
    q_max = q3+(1.5*iqr)

    return q_min,q1,q3,q_max    