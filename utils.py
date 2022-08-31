import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator as pchip
from scipy.spatial import KDTree
from dtaidistance import dtw
import scipy.io as sio
import h5py
from sklearn.metrics import mean_squared_error
import os

UI_STEP = 0.01
MIN_V = 3.2
MAX_V = 3.9 # 3.7
SIZE = 128
MIN_VAL_Q = 0
MIN_VAL_T = 0
MAX_VAL_Q = 7
MAX_VAL_T = 1

# --------------------------------------------------READ DATA--------------------------------------------------
def read_mat(file_name):
    '''
    Reads a .mat file and returns the data as a numpy array

    Parameters
	----------
	file_name: str, path to the .mat file
    '''

    return sio.loadmat(file_name)

# Function that 
def get_file_names(folder):
    '''
    Returns a list with the names of the files of a given folder

    Parameters
	----------
	folder: str, path to the folder
    '''

    file_names = []
    for file in os.listdir(folder):
        file_names.append(file)
    return file_names
# -------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------INDEXES---------------------------------------------------
def get_indexes(info, i, last_index, degradation_mode, resolution, max_degradation):
    '''
    Filters indexes according to a given resolution in a given path

    Parameters
    ----------
    info: numpy array, contains the labels for each curve
    i: int, index of the first sample in the path
    last_index: int, index of the last sample in the path
    degradation_mode: int, 0 for LLI, 1 for LAMPE, 2 for LAMNE
    resolution: int, resolution of the data
    max_degradation: int, maximum degradation of the path

    Returns
    -------
    indexes: numpy array, contains the resulting indexes
    '''

    if i < 0:
        i = 0
    # take only the path data
    path = info[i:last_index]
    # from the path data take the LLI/LAMPE/LAMNE values from 0 to 80 with a resolution of 'resolution'%
    path = KDTree(path[:,degradation_mode].reshape(path.shape[0], 1))
    res_values = np.arange(0, max_degradation, resolution)
    res_values = res_values.reshape(res_values.shape[0], 1)
    # indexes contains the indexes of the LLI/LAMPE/LAMNE values closest to the requested resolution 
    _, indexes = path.query(res_values, k=1)
    # get their global indexes inside the info matrix
    indexes = indexes + i
    return indexes

def get_curves(info, resolution, max_degradation=49.5):
    '''
    Filters indexes according to a given resolution

    Parameters
    ----------
    info: numpy array, contains the labels for each curve
    resolution: int, resolution of the data
    max_degradation: int, maximum degradation of the path

    Returns
    -------
    numpy array, contains the resulting indexes
    '''

    # retrieve indices of the paths according to the main degradation
    indexes_LLI = np.where(info[:,0] == max_degradation)[0]
    indexes_LAMPE = np.where(info[:,1] == max_degradation)[0]
    indexes_LAMNE = np.where(info[:,2] == max_degradation)[0]
    # array to save the selected indexes
    selected_indexes = []
    len_paths = 137

    # loop over the paths
    for i_LLI, i_LAMPE, i_LAMNE in zip(indexes_LLI, indexes_LAMPE, indexes_LAMNE):
        # get the indexes of the path
        indexes = get_indexes(info, i_LLI-len_paths, i_LLI, 0, resolution, max_degradation)
        # add the indexes to the array
        selected_indexes.append(indexes)

        indexes = get_indexes(info, i_LAMPE-len_paths, i_LAMPE, 1, resolution, max_degradation)
        selected_indexes.append(indexes)

        indexes = get_indexes(info, i_LAMNE-len_paths, i_LAMNE, 2, resolution, max_degradation)
        selected_indexes.append(indexes)
            
    return  np.sort(np.array(selected_indexes).flatten())
# -------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------IC DATA--------------------------------------------------
def IC(u, q, ui_step=0.0005, minV=3.2, maxV=3.9):
    '''
    Get the ICA data for a given voltage curve

    Parameters
    ----------
    u: numpy array, voltage curve
    q: numpy array, capacity curve
    ui_step: float, step of interpolation
    minV: float, minimum voltage of the IC curve
    maxV: float, maximum voltage of the IC curve

    Returns
    -------
    ui, dqi: numpy arrays, interpolated voltage and derivative of capacity
    '''

    # voltages values for which capacity is interpolated
    ui = np.arange(minV, maxV, ui_step) 
    qi = np.interp(ui, u, q)
    return ui[1:], np.diff(qi)

def reduce_size(ui, dqi, size):
    '''
    Reduces the length of the IC data to a given size

    Parameters
    ----------
    ui: numpy array, voltage curve
    dqi: numpy array, derivative of capacity (IC)
    size: int, size at which to reduce the IC data

    Returns
    -------
    numpy array, reduced IC
    '''

    curve = pchip(ui, dqi)
    ui_reduced = np.linspace(min(ui), max(ui), size)
    return curve(ui_reduced)

def normalise_data(data, type, min_val=0, max_val=7, low=0, high=1):
    '''
    Normalises the data to the range [low, high]

    Parameters
    ----------
    data: numpy array, data to normalise
    min: float, minimum value of data
    max: float, maximum value of data
    low: float, minimum value of the range
    high: float, maximum value of the range

    Returns
    -------
    normalised_data: float, normalised data
    '''
    if type == 'Q':
        min_val = MIN_VAL_Q
        max_val = MAX_VAL_Q
    elif type == 'T':
        min_val = MIN_VAL_T
        max_val = MAX_VAL_T
    normalised_data = (data - min_val)/(max_val - min_val)
    normalised_data = (high - low)*normalised_data + low
    return normalised_data

def get_IC_samples(V, Q, ui_step, minV, maxV, size):
    '''
    Returns the IC samples for each curve

    Parameters
    ----------
    V: numpy array, voltage curve
    Q: array, capacity percentages from 0 to 100 from the simulated dataset
    ui_step: float, step of the interpolation
    minV: float, minimum voltage of the IC
    maxV: float, maximum voltage of the IC
    size: int, size at which to reduce the IC data

    Returns
    -------
    info_ICs, ICs: numpy arrays, contains the labels for each curve and the IC samples
    '''

    samples = []
    for i, curve in enumerate(V):
        ui, dqi = IC(curve, Q, ui_step, minV, maxV) # ICcalc(curve, Q, ui_step, minV, maxV)
        # this means that the interpolation results in some nans
        if np.all(dqi>=0) == False:
            print("Error in curve", str(i))
            break
        else:
            new_sample = reduce_size(ui, dqi, size)
            samples.append(new_sample)
    return np.array(samples)
# -------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------TRAINING DATA--------------------------------------------
def get_data(V, Q, ui_step, minV, maxV, size, type):
    '''
    Returns training data

    Parameters
    ----------
    V: numpy array, voltage curves
    V_reference: numpy array, voltage curve of the reference cell (cycle 0)
    Q: array, capacity percentages from 0 to 100 from the simulated dataset / times
    ui_step: float, step of interpolation
    minV: float, minimum voltage of the IC
    maxV: float, maximum voltage of the IC
    size: int, size at which to reduce the IC data

    Returns
    -------
    x, y: numpy arrays, contain the IC samples and the labels for each curve
    '''
    ICs = get_IC_samples(V, Q, ui_step, minV, maxV, size)
    # normalise ICs
    #ICs = normalise_data(ICs, np.min(ICs), np.max(ICs))
    ICs = normalise_data(ICs, type)
    return ICs

def get_DTWImages(data, reference, size):
    '''
    Converts IC samples to DTW images

    Parameters
    ----------
    data: numpy array, contains the IC samples
    reference: numpy array, contains the IC of the reference cell
    size: int, size at which to reduce the IC data -> resolution of the resulting image

    Returns
    -------
    x: numpy array, DTW images
    '''

    images = []
    for IC_sample in data:
        #d, paths = dtw.warping_paths(reference, IC_sample, window=int(size/2), psi=2)
        d, paths = dtw.warping_paths(reference, IC_sample, window=size, psi=2)
        images.append(paths)
    x = np.array(images)
    # mask values that are not filled
    x = np.where(x == np.inf, -99, x)
    # negative values are replaced by 0
    x = np.where(x < 0, 0, x)
    # normalise values
    x = x/np.max(x)
    # reshape the array
    x = np.expand_dims(x, -1).astype("float32")
    return x

def save_data(path, resolution):
    '''
    Save data to disk

    Parameters
    ----------
    path: str, path to the folder where to save the data
    resolution: int, resolution of data
    '''
    
    # Load data
    QvsV = np.load('data/train/QvsV.npy', allow_pickle=True).item()
    TvsV = np.load('data/train/TvsV.npy', allow_pickle=True).item()
    info = np.load('data/train/info.npy', allow_pickle=True)
    info = np.where(info < 0, 0, info) # replace negative values by 0

    # 1. Select curves according to a given resolution
    selected_indexes = get_curves(info, resolution)
    # add the index corresponding to the reference curve
    selected_indexes = np.insert(selected_indexes, 0, 0, axis=0)
        
    x_Q = get_data(QvsV['V'][selected_indexes], QvsV['Q'], UI_STEP, MIN_V, MAX_V, SIZE-1, 'Q')
    x_T = get_data(TvsV['V'][selected_indexes], TvsV['T'], UI_STEP, MIN_V, MAX_V, SIZE-1, 'T')
    y = info[selected_indexes, 0:3]/100

    np.save(path+"/x_train_Q.npy", x_Q)
    np.save(path+"/x_train_T.npy", x_T)
    np.save(path+"/y_train.npy", y)

def save_data_new(path):
    '''
    Save data to disk

    Parameters
    ----------
    path: str, path to the folder where to save the data
    resolution: int, resolution of data
    '''

    # Load data
    QvsV = np.load('data/train/QvsV.npy', allow_pickle=True).item()
    TvsV = np.load('data/train/TvsV.npy', allow_pickle=True).item()
    info = np.load('data/train/info.npy', allow_pickle=True)
    info = np.where(info < 0, 0, info)[:, 0:3]/100 # replace negative values by 0
    y_test = np.load('data/train/y_test.npy', allow_pickle=True)

    # take only the path data
    # from the path data take the LLI/LAMPE/LAMNE values from 0 to 80 with a resolution of 'resolution'%
    path_tree = KDTree(info)
    # indexes contains the indexes of the LLI/LAMPE/LAMNE values closest to the requested resolution 
    _, selected_indexes = path_tree.query(y_test, k=1)
    # add the index corresponding to the reference curve
    selected_indexes = np.insert(selected_indexes, 0, 0, axis=0)
        
    x_Q = get_data(QvsV['V'][selected_indexes], QvsV['Q'], UI_STEP, MIN_V, MAX_V, SIZE-1)
    x_T = get_data(TvsV['V'][selected_indexes], TvsV['T'], UI_STEP, MIN_V, MAX_V, SIZE-1)
    y_train = info[selected_indexes]

    np.save(path+"/x_train_Q.npy", x_Q)
    np.save(path+"/x_train_T.npy", x_T)
    np.save(path+"/y_train.npy", y_train)

def save_DTW_data(path):
    '''
    Saves DTW data to disk

    Parameters
    ----------
    path: str, path to the folder where to save the data
    '''
    for mode in ['Q', 'T']:
        # ICs
        x = np.load(path+"/x_train_"+mode+".npy")

        # The reference IC is the first sample
        IC_reference = x[0]
        
        # Generate the DTW images
        x_DTW = get_DTWImages(x, IC_reference, SIZE)

        # Save data
        np.save(path+"/x_train_"+mode+"_DTW.npy", x_DTW)


# ----------------------------------------------------INFERENCE------------------------------------------------
def get_pred(model, x_test, y_test, path=None):
    predictions = model.predict(x_test)

    if path != None:
        # save predictions to disk as a csv file
        np.savetxt(path+'_predictions.csv', predictions, delimiter=',')

    rmse_LLI = mean_squared_error(y_test[:,0], predictions[:,0], squared=False)*100
    rmse_LAMPE = mean_squared_error(y_test[:,1], predictions[:,1], squared=False)*100
    rmse_LAMNE = mean_squared_error(y_test[:,2], predictions[:,2], squared=False)*100

    p_LLI = np.corrcoef(y_test[:,0], predictions[:,0])[0,1]
    p_LAMPE = np.corrcoef(y_test[:,1], predictions[:,1])[0,1]
    p_LAMNE = np.corrcoef(y_test[:,2], predictions[:,2])[0,1]

    return rmse_LLI, rmse_LAMPE, rmse_LAMNE, p_LLI, p_LAMPE, p_LAMNE

def plot_capacity_evolution(cycles, capacity_evolution, y_lim):
	'''
	Plots the capacity evolution of the battery.

	Parameters
	----------
	cycles: array, RPT measures of the cell
    capacity_evolution: array, capacity evolution of the cell
	y_lim: int, the limits of the y axis (percentage of capacity)
	'''
	
	plt.scatter(cycles, capacity_evolution, marker="^", s=100)
	plt.plot(cycles, capacity_evolution)
	plt.grid()
	plt.ylim(y_lim, 100)
	plt.xlabel('Cycle #')
	plt.ylabel('Normalized capacity (%)')
	plt.show()

def get_capacity_prediction(info, predictions):
	'''
	Gets capacity from predictions of the degradation modes

	Parameters
	----------
	info: dictionary, contains the information about the training set
	predictions: array, the predictions of the model

	Returns
	-------
	capacity_prediction: array, capacity predictions
	'''
	from scipy.spatial import KDTree

	capacity_prediction = []
	for prediction in predictions:
		path = KDTree(info[:,0:3])
		_, index = path.query(prediction*100, k=1)
		capacity_prediction.append(100-info[index][3])

	return capacity_prediction

def plot_predictions(cycles, predictions, capacity_evolution, capacity_prediction, y_lim):
	'''
	Plots the predictions of the model

	Parameters
	----------
	cycles: array, cycles of the real cell
    predictions: array, predictions of the model
    capacity_evolution: array, capacity evolution of the real cell
    capacity_prediction: array, capacity predictions
    y_lim: float, the y limit of the plot
	'''
	# Degradation modes
	plt.plot(cycles, predictions[:,0]*100, label='LLI')
	plt.plot(cycles, predictions[:,1]*100, label='LAMPE')
	plt.plot(cycles, predictions[:,2]*100, label='LAMNE')
	plt.xlabel('Cycle #')
	plt.ylabel('Predicted degradation (%)')
	plt.legend()
	plt.show()

	# Capacity
	plt.scatter(cycles, capacity_evolution)
	plt.plot(cycles, capacity_evolution)
	plt.scatter(cycles, capacity_prediction)
	plt.plot(cycles, capacity_prediction)
	plt.ylim(y_lim, 100)
	plt.xlabel('Cycle #')
	plt.ylabel('Normalized capacity (%)')
	plt.show()