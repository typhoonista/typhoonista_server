import pickle
import numpy as np
import pandas as pd

boeke_model_regr_rf = pickle.load(open('STACKED-SVR_RF-DYIELD.joblib', 'rb'))
print(boeke_model_regr_rf)