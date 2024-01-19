import pickle
import numpy as np
import pandas as pd

boeke_model_regr_rf = pickle.load(open('models\STACKED-SVR_RF102.joblib', 'rb'))
print(boeke_model_regr_rf)