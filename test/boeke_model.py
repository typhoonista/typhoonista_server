import pickle
import numpy as np
import pandas as pd

boeke_model_regr_rf = pickle.load(open('STACKED-SVR_RF_FINAL.joblib', 'rb'))
boeke_model_regr_rf1 = pickle.load(open('model_boeke.sav', 'rb'))
print(boeke_model_regr_rf)
print(boeke_model_regr_rf1)