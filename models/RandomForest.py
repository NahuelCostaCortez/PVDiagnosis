from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

class RandomForest:
    def __init__(self, criterion, max_depth, n_estimators):
        self.model = RandomForestRegressor(max_depth, criterion, n_estimators)
        
    def fit(self, X, y):
        self.model.fit(X, y)

    def save_model(self, path):
        dump(self.model, path+'.joblib')

    def load_model(self, path):
        self.model = load(path+'.joblib')

    def predict(self, X):
        return self.model.predict(X)