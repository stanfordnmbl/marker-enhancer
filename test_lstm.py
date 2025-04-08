import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import dataman
import copy
import tensorflow as tf

from settings import get_settings_lstm
from utilities import TRC2numpy
from utilities import getMarkersPoseDetector_lowerExtremity, getMarkersAugmenter_lowerExtremity
from utilities import getMarkersPoseDetector_upperExtremity, getMarkersAugmenter_upperExtremity

# %% User settings.
# Set to True if you want to test the reference model. If set to False, then
# specify the case you want to test in the variable 'case'.
test_reference_model = True
# Select case you want to test, see settings.
case = '' # ignored if test_reference_model is True

# Options are 'drop_vertical_jump', 'running', 'squats'
# The files are in the test_data folder
test_marker_filename = 'drop_vertical_jump'
# TODO
subject_height = 1.96
subject_weight = 78.2

# %% Paths.
path_main = os.getcwd()
path_trc_file = os.path.join(path_main, 'test_data', test_marker_filename + '.trc')
path_trc_file_out = os.path.join(path_main, 'test_data', 'enhanced', 'lstm')
if test_reference_model:
    path_models = os.path.join(path_main, 'reference_models', 'lstm', 'body') # TODO
    path_trc_file_out = os.path.join(path_trc_file_out, 'reference_model')
else:
    path_models = os.path.join(path_main, 'trained_models', 'lstm', 'body', case) # TODO
    path_trc_file_out = os.path.join(path_trc_file_out, case)
os.makedirs(path_trc_file_out, exist_ok=True)

# %% Settings
# TODO
case = 'body_reference'
settings = get_settings_lstm(case)
augmenter_type = settings['augmenter_type']
pose_detector = settings['poseDetector']
reference_marker = 'midHip'
if augmenter_type == 'lowerExtremity':    
    feature_markers = getMarkersPoseDetector_lowerExtremity(pose_detector)[0]
    response_markers = getMarkersAugmenter_lowerExtremity()[0]
elif augmenter_type == 'upperExtremity':    
    feature_markers = getMarkersPoseDetector_upperExtremity(pose_detector)[0]
    response_markers = getMarkersAugmenter_upperExtremity()[0]
feature_markers_raw = [x.split('_')[0] for x in feature_markers]

# %% Prepare inputs.
# Step 1: import .trc file with OpenPose marker trajectories.
trc_data = TRC2numpy(path_trc_file, feature_markers_raw)
trc_data_time = trc_data[:,0]
trc_data_data = trc_data[:,1:]

# Step 2: Normalize with reference marker position.
trc_file = dataman.TRCFile(path_trc_file)
time = trc_file.time
referenceMarker_data = trc_file.marker(reference_marker)  
norm_trc_data_data = np.zeros((trc_data_data.shape[0],trc_data_data.shape[1]))
for i in range(0,trc_data_data.shape[1],3):
    norm_trc_data_data[:,i:i+3] = trc_data_data[:,i:i+3] - referenceMarker_data
    
# Step 3: Normalize with subject's height.
norm2_trc_data_data = copy.deepcopy(norm_trc_data_data)
norm2_trc_data_data = norm2_trc_data_data / subject_height

# Step 4: Add remaining features.
inputs = copy.deepcopy(norm2_trc_data_data)
inputs = np.concatenate((inputs, subject_height*np.ones((inputs.shape[0],1))), axis=1)
inputs = np.concatenate((inputs, subject_weight*np.ones((inputs.shape[0],1))), axis=1)
    
# %% Predict outputs
mean_subtraction = settings['mean_subtraction'] 
std_normalization = settings['std_normalization'] 

# Import trained model and weights
json_file = open(os.path.join(path_models, 'model.json'), 'r')
pretrainedModel_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(pretrainedModel_json)  

# Load weights into new model.
model.load_weights(path_models, 'weights.h5')

# Process data
if mean_subtraction:
    trainFeatures_mean = np.load(os.path.join(path_models, 'mean.npy'), allow_pickle=True)
    inputs -= trainFeatures_mean
if std_normalization:
    trainFeatures_std = np.load(os.path.join(path_models, 'std.npy'), allow_pickle=True)
    inputs /= trainFeatures_std
outputs = model.predict(np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1])), verbose=0)
outputs = np.reshape(outputs, (outputs.shape[1], outputs.shape[2]))

# %% Post-process outputs    
# Step 1: Un-normalize with subject's height.
unnorm_outputs = outputs * subject_height

# Step 2: Un-normalize with reference marker position.
unnorm2_outputs = np.zeros((unnorm_outputs.shape[0],unnorm_outputs.shape[1]))
for i in range(0,unnorm_outputs.shape[1],3):
    unnorm2_outputs[:,i:i+3] = unnorm_outputs[:,i:i+3] + referenceMarker_data
    
# %% Return augmented .trc file
for c, marker in enumerate(response_markers):
    x = unnorm2_outputs[:,c*3]
    y = unnorm2_outputs[:,c*3+1]
    z = unnorm2_outputs[:,c*3+2]
    trc_file.add_marker(marker, x, y, z)
pathFileOut = os.path.join(path_trc_file_out, '{}_enhanced.trc'.format(test_marker_filename))
trc_file.write(pathFileOut)
