# PVDiagnosis

Photovoltaic batteries are often used sporadically, which prevents the application of traditional diagnostic methods. This repository contains the code for training and testing machine learning diagnosis algorithms on photovoltaic battery charging data.
The training data corresponds to synthetic voltage data under different degradations calculated from clear sky model irradiance data. Testing data corresponds to synthetic voltage responses calculated from plane of array irradiance
observations.

The data will be made public soon. 


# Files in this Repository
- \data: samples with which to train and evaluate the models.
- \models: folder containing the different models.
- mat_to_npy.py: the original data format is .mat. This file converts and processes the data to a .npy format
- train.py: models training.
- utils.py: helper functions.
- evaluateCloudy.py: script to evaluate diagnosis on cloudy data.
- evaluateSunny.py: script to evaluate diagnosis on sunny data.
- requirements.txt: requirements for the project.

To execute the python code we recommend setting up a new python environment with packages matching the requirements.txt file. It can be easily done with anaconda: conda create --name --file requirements.txt.