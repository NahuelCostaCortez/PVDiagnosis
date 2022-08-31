import utils
import os
import numpy as np

path = './data/test/'
test_folders = ['MEDB_Irr1', 'MEDB_IrrM1', 'MEDB_Cloud1']

for folder in test_folders:
    files = utils.get_file_names(path+folder)
    npy_files = [name for name in files if name.endswith(".npy")]
    # delete all npy files in the folder
    for name in npy_files:
        os.remove(path+folder+'/'+name)

    files = utils.get_file_names(path+folder)
    y_test = utils.read_mat(path+folder+'/'+files[0])['pathinfo'][:, 0:3]/100
    np.save(path+folder+"/y_test.npy", y_test)
    
    for file in files:
        type = file.split('.mat')[0].split('_')[-1]
        
        ICs = []
        ICsDTW = []
        if type == 'V':
            data = utils.read_mat(path+folder+'/'+file)
            ICs = utils.get_data(data['volt'].T, data['Q'].flatten(), utils.UI_STEP, utils.MIN_V, utils.MAX_V, utils.SIZE-1, 'Q')
            ICsDTW = utils.get_DTWImages(ICs, ICs[0], utils.SIZE)
        if type == 't':
            data = utils.read_mat(path+folder+'/'+file)
            ICs = utils.get_data(data['voltT'].T, data['timenorm'].flatten(), utils.UI_STEP, utils.MIN_V, utils.MAX_V, utils.SIZE-1, 'T')
            ICsDTW = utils.get_DTWImages(ICs, ICs[0], utils.SIZE)

        np.save(path+folder+'/'+file.split('.mat')[0], ICs)
        np.save(path+folder+'/'+file.split('.mat')[0]+"_DTW.npy", ICsDTW)