import numpy as np
import utils
from models import RandomForest, XGBoost, FFN, CNN_1D, CNN_DTW
import pandas as pd

# ------------- DATA LOADING --------------
size = 128
path_saved = './saved/'
path_test = './data/test/'
test_folders = ['MEDB_Full1', 'MEDB_Irr1', 'MEDB_IrrM1']
models = ['RF', 'XGB', 'FFN', 'CNN_1D', 'CNN_DTW']

# Iterate over test folders
for folder in test_folders:
    print(folder)
    y_test = np.load(path_test+folder+'/y_test.npy')
    # Get models depending on mode - Q/T
    for mode in ["Q","T"]:

        # Evaluate for each file
        files = utils.get_file_names(path_test+folder)
        type = 'V' if mode == 'Q' else 't'
        npy_files = [name for name in files if name.endswith("_"+type+".npy")]
        print(files)

        df_RMSE = pd.DataFrame(index=['LLI', 'LAMPE', 'LAMNE', 'Mean'])
        for model in models:
            preds = []
            if model == 'RF':
                model_obj = RandomForest.RandomForest(criterion="squared_error", max_depth=100, n_estimators=2000)
                model_obj.load_model(path_saved+"model-RF_"+mode)
            elif model == 'XGB':
                model_obj =  XGBoost.XGBoost('reg:squarederror', max_depth=10, eta=0.2)
                model_obj.load_model(path_saved+"model-XGB_"+mode)
            elif model == 'FFN':
                model_obj = FFN.FFN(size-1, 64, 0.001)
                model_obj.load_model(path_saved+"model-FFN_"+mode)
            elif model == 'CNN_1D':
                model_obj = CNN_1D.CNN_1D(size-1, 32, 0.01)
                model_obj.load_model(path_saved+"model-CNN_1D_"+mode)
            elif model == 'CNN_DTW':
                model_obj = CNN_DTW.CNN_DTW(size, 32, 0.01)
                model_obj.load_model(path_saved+"model-CNN_DTW_"+mode)
                npy_files = [name for name in files if name.endswith(type+"_DTW.npy")]
            else:
                print('Model not found')
            
            # remove previous results from folder
            #os.system('rm -r '+"./results/"+folder+'/*')
            for file_npy in npy_files:
                print("\t"+file_npy)

                x_test = np.load(path_test+folder+'/'+file_npy)
                if model == 'CNN_1D':
                    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

                # get results
                rmse_LLI, rmse_LAMPE, rmse_LAMNE, p_LLI, p_LAMPE, p_LAMNE = utils.get_pred(model_obj, x_test, y_test, "./results/"+folder+'/'+mode+'/'+model+"_"+file_npy.split('.npy')[0])
                pred_RMSE = [rmse_LLI, rmse_LAMPE, rmse_LAMNE]
                # append pred_RMSE to preds
                preds.append(pred_RMSE)
            final_preds = np.array(preds).mean(axis=0)
            df_RMSE[model] = np.hstack((final_preds, np.array(final_preds).mean(axis=0)))
        print("Saving results as: ./results/"+folder+'/'+mode+"/summary_results_RMSE.csv\n")
        df_RMSE.to_csv("./results/"+folder+'/'+mode+"/summary_results_RMSE.csv")