import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(path):
    df = pd.read_csv(path)

    cols_to_drop = ['filename', 'Unnamed: 0', 'spec_bw', 'rmse', 'mfcc5']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    X = df.drop('raga', axis=1).values
    y = df['raga'].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = X.reshape(X.shape[0], 1, X.shape[1])

    return train_test_split(X, y, test_size=0.2, stratify=y), encoder