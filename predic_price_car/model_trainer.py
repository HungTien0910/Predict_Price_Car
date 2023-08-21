import xgboost as xgb
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np

class CarModelTrainer:
    def __init__(self,feature_columns):
        self.model = None
        self.feature_columns = feature_columns
    def train_model(self, X_train, y_train):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'random_state': 42
        }
        num_boost_round = 100
        self.model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    def evaluate_model(self, X_test, y_test):
        dtest = xgb.DMatrix(X_test)
        y_pred = self.model.predict(dtest)
        r2 = r2_score(y_test, y_pred)
        return r2

    def predict_price(self, new_car_data):
        new_car_df = pd.DataFrame(new_car_data, columns=self.feature_columns)
        return self.model.predict(xgb.DMatrix(new_car_df))[0]