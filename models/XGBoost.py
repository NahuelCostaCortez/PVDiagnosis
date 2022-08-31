from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from joblib import dump, load

class XGBoost:
    def __init__(self, criterion, max_depth, eta):
        self.model = MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', max_depth=max_depth, eta=eta))
        
    def fit(self, X, y):
        self.model.fit(X, y)

    def save_model(self, path):
        dump(self.model, path+'.joblib')

    def load_model(self, path):
        self.model = load(path+'.joblib')

    def predict(self, X):
        return self.model.predict(X)