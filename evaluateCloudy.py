import numpy as np
import utils
from models import RandomForest, XGBoost, FFN, CNN_1D, CNN_DTW
import pandas as pd

# ------------- DATA LOADING --------------
size = 128
path_saved = './saved/'
path_test = 'data/test/'
folder = 'MEDB_Cloud1'
models = ['RF', 'XGB', 'FFN', 'CNN_1D', 'CNN_DTW']

# training cells
foldersA_saved = ["38","39","40","47","48","49"]
# testing cells
foldersA_result = ["30","31","29","56","57","58"]

foldersB_saved = ["41","42","43","50","51","52"]
foldersB_result = ["33","34","32","59", "60","61"]

foldersC_saved = ["44","45","46","53","54","55"]
foldersC_result = ["35","37","36","62","63","64"]



path_saved = "./saved/cloudy/"
mode = 'T'
size = 128
y_test = np.load(path_test+folder+'/y_test.npy')

for mode in ["Q","T"]:
    type = 'V' if mode == 'Q' else 't'
    for folders in [zip(foldersA_saved, foldersA_result, ["A"]*len(foldersA_saved)), zip(foldersB_saved, foldersB_result, ["B"]*len(foldersB_saved)), zip(foldersC_saved, foldersC_result, ["C"]*len(foldersC_saved))]:
        for saved, result, folder_save in folders:
            df_RMSE = pd.DataFrame(index=['LLI', 'LAMPE', 'LAMNE', 'Mean'])
            for model in ["RF", "XGB", "FFN", "CNN_1D", "CNN_DTW"]:
                if model == 'RF':
                    model_obj = RandomForest.RandomForest(criterion="squared_error", max_depth=100, n_estimators=2000)
                    model_obj.load_model(path_saved+saved+"/model-RF_"+mode)
                elif model == 'XGB':
                    model_obj =  XGBoost.XGBoost('reg:squarederror', max_depth=10, eta=0.2)
                    model_obj.load_model(path_saved+saved+"/model-XGB_"+mode)
                elif model == 'FFN':
                    model_obj = FFN.FFN(size-1, 64, 0.001)
                    model_obj.load_model(path_saved+saved+"/model-FFN_"+mode)
                elif model == 'CNN_1D':
                    model_obj = CNN_1D.CNN_1D(size-1, 32, 0.01)
                    model_obj.load_model(path_saved+saved+"/model-CNN_1D_"+mode)
                elif model == 'CNN_DTW':
                    model_obj = CNN_DTW.CNN_DTW(size, 32, 0.01)
                    model_obj.load_model(path_saved+saved+"/model-CNN_DTW_"+mode)
                else:
                    print('Model not found')
                if model == 'RF' or model == 'XGB' or model == 'FFN' or model == 'CNN_1D':
                    x_test = np.load("./data/test/MEDB_Cloud1/"+result+"_"+type+".npy")
                    if model == 'CNN_1D':
                        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
                if model == 'CNN_DTW':
                    x_test = np.load("./data/test/MEDB_Cloud1/"+result+"_"+type+"_DTW.npy")
                    

                rmse_LLI, rmse_LAMPE, rmse_LAMNE, p_LLI, p_LAMPE, p_LAMNE = utils.get_pred(model_obj, x_test, y_test, "./results/MEDB_Cloud1/"+folder_save+"/"+result+"/"+mode+"/"+model)
                pred_RMSE = [rmse_LLI, rmse_LAMPE, rmse_LAMNE]
                df_RMSE[model] = np.hstack((pred_RMSE , np.array(pred_RMSE).mean(axis=0)))
            print("Saving results as: ./results/MEDB_Cloud1/"+folder_save+"/"+result+'/'+mode+"/summary_results_RMSE.csv\n")
            df_RMSE.to_csv("./results/MEDB_Cloud1/"+folder_save+"/"+result+'/'+mode+"/summary_results_RMSE.csv")