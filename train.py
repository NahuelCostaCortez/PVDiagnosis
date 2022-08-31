'''
Training of the selected models. 
No hyperparameter tuning is performed as they were implemented as stated in their corresponding papers.
'''

import numpy as np
from models import RandomForest, CNN_1D, FFN, XGBoost

# ------------- DATA LOADING --------------
size = 128
path = "data/train"

x_train_T = np.load(path+"/x_train_T.npy")
x_train_Q = np.load(path+"/x_train_Q.npy")
y_train = np.load(path+"/y_train.npy")

for x_train, mode in zip([x_train_T, x_train_Q], ["T", "Q"]):
    # ------------- RANDOM FOREST -------------
    model = RandomForest.RandomForest(criterion="mse", max_depth=100, n_estimators=2000)
    model.fit(x_train, y_train)
    model.save_model("./saved/model-RF_"+mode)

    # ----------------- XGB -----------------
    model = XGBoost.XGBoost('reg:squarederror', max_depth=10, eta=0.2)
    model.fit(x_train, y_train)
    model.save_model("./saved/model-XGB_"+mode)

    # ------------------- FFN -------------------
    model = FFN.FFN(size-1, 64, 0.001)
    model.fit(x_train, y_train)
    model.save_model("./saved/model-FFN_"+mode)

    # ----------------- CNN_1D -----------------
    model = CNN_1D.CNN_1D(size-1, 32, 0.01)
    model.fit(np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)), y_train)
    model.model.load_weights('./checkpoints/checkpoint')
    model.save_model("./saved/model-CNN_1D_"+mode)

    # ----------------- CNN_DTW -----------------
    model = CNN_DTW.CNN_DTW(size, 32, 0.01)
    if mode == 'T':
        x_train = np.load("./data/train/x_train_T_DTW.npy")
    else:
        x_train = np.load("./data/train/x_train_Q_DTW.npy")
    model.fit(x_train, y_train)
    model.save_model("./saved/model-CNN_DTW_"+mode)