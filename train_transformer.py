'''
    ---------------------------------------------------------------------------
    train_transformer.py
    ---------------------------------------------------------------------------
    Copyright 2022 Stanford University and the Authors
    
    Author(s): Antoine Falisse

    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pickle
import platform
import multiprocessing
import tensorflow as tf
import json
import h5py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from settings import get_settings_transformer
from data_generator import dataGenerator
from datasets import getInfodataset
from utilities import getPartition, get_partition_name, get_mean_name
from utilities import rotateArray, rotateArraySphere4, get_idx_in_all_features, get_reference_marker_value
from utilities import subtract_reference_marker_value, normalize_height
from utilities import get_circle_rotation, get_noise, get_height, getInfoDataName, getResampleName
from utilities import getMarkersPoseDetector_lowerExtremity, getMarkersAugmenter_lowerExtremity
from utilities import getMarkersPoseDetector_upperExtremity, getMarkersAugmenter_upperExtremity
from transformer_model import get_transformer_encoderonly_model
from transformer_model import  Augmenter_EncoderOnly, ExportAugmenter_EncoderOnly

# %% User inputs.
# Select case you want to train, see mySettings for case-specific settings.
cases = ["body_example", "arm_example"]
model_type = 'transformer'

runTraining = True
saveTrainedModel = True
inferenceTesting = False

for case in cases:   

    # %% Load settings.
    print("Training case: {}".format(case))
    settings = get_settings_transformer(case)
    augmenter_type = settings["augmenter_type"]
    poseDetector = settings["poseDetector"]
    idxDatasets = settings["idxDatasets"]
    scaleFactors = settings["scaleFactors"]
    nEpochs = settings["nEpochs"]
    batchSize = settings["batchSize"]
    mean_subtraction = settings["mean_subtraction"]
    std_normalization = settings["std_normalization"]

    # Number of layers.
    nHLayers = 6 # Default from ref paper
    if "nHLayers" in settings:
        nHLayers = settings["nHLayers"]

    # Number of multi-head attention heads.
    nHeads = 8 # Default from ref paper
    if "nHeads" in settings:
        nHeads = settings["nHeads"]

    # Dimensionality of the embedding space.
    d_model = 512 # Default from ref paper
    if "d_model" in settings:
        d_model = settings["d_model"]

    # Dimensionality of the feed-forward network.
    d_ff = 2048 # Default from ref paper
    if "d_ff" in settings:
        d_ff = settings["d_ff"]

    # Dropout.
    dropout = 0.1 # Default from ref paper
    if "dropout" in settings:
        dropout = settings["dropout"]

    # WIP: attention_axes.
    attention_axes = None
    if "attention_axes" in settings:
        attention_axes = settings["attention_axes"]
        
    # Auto-regressive decoder.
    autoregressive = False
    if "autoregressive" in settings:
        autoregressive = settings["autoregressive"]

    # Encoder only.
    encoder_only = False
    if "encoder_only" in settings:
        encoder_only = settings["encoder_only"]
        
    # Learning rate.
    learning_r = 'custom'
    if "learning_r" in settings:
        learning_r = settings["learning_r"]
        
    # Loss function.
    loss_f = "mean_squared_error" # default
    if "loss_f" in settings:
        loss_f = settings["loss_f"]
        
    # Noise.
    noise_bool = False # default
    noise_magnitude = 0 # default
    noise_type = '' # default
    if "noise_magnitude" in settings:
        noise_magnitude = settings["noise_magnitude"]
        if noise_magnitude > 0:
            noise_bool = True
        noise_type = 'per_timestep'
        if 'noise_type' in settings:
            noise_type = settings["noise_type"]
            
    # Rotation.
    withRotation = False
    if 'withRotation' in settings:
        withRotation = settings["withRotation"]
    nRotations = 1
    if "nRotations" in settings:
        nRotations = settings["nRotations"]    
    rotation_type = 'circleRotation'
    if "rotation_type" in settings:
        rotation_type = settings["rotation_type"]
    mixedCircleSphereRotations = {}
    if rotation_type == 'mixedCircleSphereRotation':
        mixedCircleSphereRotations = settings["mixedCircleSphereRotations"]
        nCircleRotations = mixedCircleSphereRotations['nCircleRotations']
        nSphereRotations = mixedCircleSphereRotations['nSphereRotations']
        if nCircleRotations + nSphereRotations != nRotations:
            raise ValueError("nCircleRotations + nSphereRotations != nRotations")
        
    # Allows to select only part of the data in datasets tagged with activity name
    partial_selection = {"activity":[], "factor":[], "excluded":[]}
    if 'partial_selection' in settings:
        partial_selection = settings["partial_selection"]
    if not "excluded" in partial_selection:
        partial_selection["excluded"] = []

    # Allows to select nScaleFactors out of all scaleFactors available. That way
    # can deal with smaller dataset and avoid redundancy. If -1, then uses all
    # scaleFactors
    nScaleFactors = -1
    if 'nScaleFactors' in settings:
        nScaleFactors = settings["nScaleFactors"]

    # Reference markers.
    reference_marker = 'midHip'
    if 'reference_marker' in settings:
        reference_marker = settings["reference_marker"]

    # Marker weights.
    marker_weights = None
    if 'marker_weights' in settings:
        marker_weights = settings["marker_weights"]

    sensitivity_model = ''
    if 'sensitivity_model' in settings:
        sensitivity_model = settings["sensitivity_model"]
        
    num_frames = 60
    if 'num_frames' in settings:
        num_frames = settings["num_frames"]    
        
    h5 = True
    if 'h5' in settings:
        h5 = settings["h5"]
    if h5:
        prefixH5 = 'h5_'
    else:
        prefixH5 = ''

    curated_datasets = True
    if 'curated_datasets' in settings:
        curated_datasets = settings["curated_datasets"]
    if curated_datasets:
        prefix_curated_datasets = '_curated'
    else:
        prefix_curated_datasets = ''

    old_data = False
    prefix_old_data = ''
    idxFold = 0
        
    # Default sampling frequency is 60Hz
    default_sf = 60
    if 'sampling_frequencies' in settings:
        if 'default_sf' in settings['sampling_frequencies']:
            default_sf = settings['sampling_frequencies']['default_sf']    
    resample = {"Dataset": [i for i in idxDatasets], "fs": [default_sf for i in range(0,len(idxDatasets))]}
    if 'sampling_frequencies' in settings:
        for i in settings['sampling_frequencies']['Dataset']:
            if i in resample["Dataset"]:
                resample["fs"][resample["Dataset"].index(i)] = settings['sampling_frequencies']["fs"][settings['sampling_frequencies']["Dataset"].index(i)]

    # I am having an issue with using a lot of validation data to compute the
    # val loss during training and therefore using early stopping. Let's add
    # the option to only use part of the validation data when evaluating the
    # val loss. For instance, no need to use all five scale factors and all
    # rotations.
    different_data_val_loss = False
    if 'different_data_val_loss' in settings:
        different_data_val_loss = settings["different_data_val_loss"]
    if different_data_val_loss:
        scaleFactors_val_loss = [1.0]
        if 'scaleFactors_val_loss' in settings:
            scaleFactors_val_loss = settings["scaleFactors_val_loss"]
        nScaleFactors_val_loss = -1
        if 'nScaleFactors_val_loss' in settings:
            nScaleFactors_val_loss = settings["nScaleFactors_val_loss"]
            
        withRotation_val_loss = True
        if 'withRotation_val_loss' in settings:
            withRotation_val_loss = settings["withRotation_val_loss"]
        nRotations_val_loss = 4
        if 'nRotations_val_loss' in settings:
            nRotations_val_loss = settings["nRotations_val_loss"]
        rotation_type_val_loss = 'mixedCircleSphereRotation'
        if 'rotation_type_val_loss' in settings:
            rotation_type_val_loss = settings["rotation_type_val_loss"]
            
        mixedCircleSphereRotations_val_loss = {}
        if rotation_type_val_loss == 'mixedCircleSphereRotation':
            mixedCircleSphereRotations_val_loss = {'nCircleRotations': 3, 'nSphereRotations': 1}
            if 'mixedCircleSphereRotations_val_loss' in settings:
                mixedCircleSphereRotations_val_loss = settings["mixedCircleSphereRotations_val_loss"]                
            nCircleRotations_val_loss = mixedCircleSphereRotations_val_loss['nCircleRotations']
            nSphereRotations_val_loss = mixedCircleSphereRotations_val_loss['nSphereRotations']
            if nCircleRotations_val_loss + nSphereRotations_val_loss != nRotations_val_loss:
                raise ValueError("nCircleRotations_val_loss + nSphereRotations_val_loss != nRotations_val_loss")

    # %% Paths.
    pathMain = os.getcwd()
    pathTrainedModels = os.path.join(pathMain, 'trained_models', model_type, case)
    os.makedirs(pathTrainedModels, exist_ok=True)
        
    # %% Settings.
    featureHeight = True
    featureWeight = True
    marker_dimensions = ["x", "y", "z"]
    nDim = len(marker_dimensions)    
    normalizeDataHeight = True

    # Use multiprocessing (only working on Linux apparently).
    # https://keras.io/api/models/model_training_apis/
    if platform.system() == 'Linux':
        use_multiprocessing = True
        nWorkers = multiprocessing.cpu_count() - 16
        print('Using {} workers'.format(nWorkers))
    else:
        # Not supported on Windows
        use_multiprocessing = False
        nWorkers = 1
        
    pathData = os.path.join(pathMain, "Data")
    pathData_all = os.path.join(pathData, "training_data{}_{}_{}{}{}".format(prefix_curated_datasets, num_frames, poseDetector, prefix_old_data, sensitivity_model))
        
    # Reference vector for rotation
    ref_vec = np.array([0,0,1])

    # Remove datasets that do not have arms 
    if augmenter_type == 'upperExtremity':
        idxDatasets_noArms = []
        for idxDataset in idxDatasets:
            infoDataset = getInfodataset(idxDataset)
            if not infoDataset['arms']:
                idxDatasets_noArms.append(idxDataset)
        # Remove datasets in idxDatasets_noArms from idxDatasets
        idxDatasets = [i for i in idxDatasets if i not in idxDatasets_noArms]
    print('Number of datasets: {}'.format(len(idxDatasets)))

    # %% Helper indices.
    # Get indices features/responses based on augmenter_type and poseDetector.    
    if augmenter_type == 'lowerExtremity':
        feature_markers = getMarkersPoseDetector_lowerExtremity(poseDetector)[0]
        response_markers = getMarkersAugmenter_lowerExtremity()[0]
    elif augmenter_type == 'upperExtremity':
        feature_markers = getMarkersPoseDetector_upperExtremity(poseDetector)[0]
        response_markers = getMarkersAugmenter_upperExtremity()[0]
    nFeature_markers = len(feature_markers)*nDim
    nResponse_markers = len(response_markers)*nDim
    # Additional features (height and weight).
    nAddFeatures = 0
    if featureHeight:
        nAddFeatures += 1
    if featureWeight:
        nAddFeatures += 1
        
    # %% Process settings.
    if loss_f == 'weighted_l2_loss':
        # Create vector of weights for loss function.
        weights_loss = np.ones(nResponse_markers)
        for i, marker in enumerate(response_markers):
            if marker in marker_weights:
                weights_loss[i*nDim:(i+1)*nDim] = marker_weights[marker]
    else:
        weights_loss = None 

    # %% Partition (select data for training, validation, and test).
    datasetName = ' '.join([str(elem) for elem in idxDatasets])
    datasetName = datasetName.replace(' ', '')
    scaleFactorName = ' '.join([str(elem) for elem in scaleFactors])
    scaleFactorName = scaleFactorName.replace(' ', '')
    scaleFactorName = scaleFactorName.replace('.', '')
    partitionNameAll = ('{}_{}_{}_{}'.format(augmenter_type, datasetName,
                                        scaleFactorName, nScaleFactors))
    
    if different_data_val_loss:        
        scaleFactorName_val_loss = ' '.join([str(elem) for elem in scaleFactors_val_loss])
        scaleFactorName_val_loss = scaleFactorName_val_loss.replace(' ', '')
        scaleFactorName_val_loss = scaleFactorName_val_loss.replace('.', '')        
        partitionNameAll_val_loss = ('{}_{}_{}_{}'.format(
            augmenter_type, datasetName,
            scaleFactorName_val_loss, nScaleFactors_val_loss))    
    
    if 'activity' in partial_selection:
        if partial_selection['activity']:
            for i, activity in enumerate(partial_selection['activity']):
                partitionNameAll += ('_' + activity + str(partial_selection['factor'][i]))
                if different_data_val_loss: 
                    partitionNameAll_val_loss += ('_' + activity + str(partial_selection['factor'][i]))
    if 'excluded' in partial_selection:
        if partial_selection['excluded']:
            partitionNameAll += '_full'
            partitionNameAll_val_loss += '_full'
            for i in partial_selection['excluded']:
                partitionNameAll += str(i)
                if different_data_val_loss: 
                    partitionNameAll_val_loss += str(i)
            
    # Hacky part.
    # These names are too long to deal with. Let's generate a random name.
    partitionName = get_partition_name(partitionNameAll)
    pathPartition = os.path.join(pathData_all, 'partition_{}.npy'.format(partitionName))
    if different_data_val_loss: 
        # print(partitionNameAll_val_loss)
        partitionName_val_loss = get_partition_name(partitionNameAll_val_loss)
        pathPartition_val_loss = os.path.join(pathData_all, 'partition_{}.npy'.format(partitionName_val_loss))        

    # Load infoData dict.    
    infoDataNameAll = ''
    for idxDataset in idxDatasets:
        if int(idxDataset) in resample["Dataset"]:
            fs = resample["fs"][resample["Dataset"].index(int(idxDataset))]
            if num_frames == 60:
                if not fs == 60:
                    suffix_sf = "{}-{}fs".format(idxDataset, fs)
                else:
                    suffix_sf = ""
            elif num_frames == 30:
                if not fs == 60:
                    suffix_sf = "{}-{}fs".format(idxDataset, fs)
                else:
                    suffix_sf = ""
            elif num_frames == 120:
                if not fs == 120:
                    suffix_sf = "{}-{}fs".format(idxDataset, fs)
                else:
                    suffix_sf = ""
        else:
            suffix_sf = ""
        infoDataNameAll += suffix_sf        
    if infoDataNameAll == '':
        infoDataName = ''
    else:
        infoDataName = '_' + getInfoDataName(infoDataNameAll)
    infoData = np.load(os.path.join(pathData_all, 'infoData_{}_{}{}.npy'.format(num_frames, poseDetector, infoDataName)),
                       allow_pickle=True).item()
    print('Loading infoData_{}_{}{}.npy'.format(num_frames, poseDetector, infoDataName))

    # Load splitting settings.
    subjectSplit = np.load(os.path.join(pathData, 'subjectSplit{}{}.npy'.format(prefix_curated_datasets, prefix_old_data)),
                           allow_pickle=True).item()
    # Get partition.
    if not os.path.exists(pathPartition):
        print("Computing main partition")
        partition = getPartition(idxDatasets, scaleFactors, infoData, subjectSplit, 
                                 idxFold, partial_selection, nScaleFactors)
        np.save(pathPartition, partition)
    else:
        print("Loading main partition")
        partition = np.load(pathPartition, allow_pickle=True).item()
        
    if different_data_val_loss:
        if not os.path.exists(pathPartition_val_loss):
            print("Computing val loss partition")
            partition_val_loss = getPartition(idxDatasets, scaleFactors_val_loss, infoData, subjectSplit, 
                                     idxFold, partial_selection, nScaleFactors_val_loss)
            np.save(pathPartition_val_loss, partition_val_loss)
        else:
            print("Loading val loss partition")
            partition_val_loss = np.load(pathPartition_val_loss, allow_pickle=True).item()
        
    # Mapping contains part of infoData, the mapping between the indexes and 
    # the datasets. Goal is to deal with a smaller file.
    mapping = {'datasets': infoData['datasets'], 'indexes': infoData['indexes'],
               'datasets_arms_idx': [], 'datasets_arms_bool': []}
    for idxDataset in idxDatasets:
        infoDataset = getInfodataset(idxDataset)    
        mapping['datasets_arms_idx'].append(idxDataset)
        mapping['datasets_arms_bool'].append(infoDataset['arms'])
        
    print("# sequences train set {}".format(partition['train'].shape[0]))
    print("# sequences validation set {}".format(partition['val'].shape[0]))
    print("# sequences test set {}".format(partition['test'].shape[0]))
    if different_data_val_loss:
        print("# sequences validation set val loss {}".format(partition_val_loss['val'].shape[0]))
            
    # %% Data processing: compute mean and standard deviation per dataset and then average.
    meanNameAll = '{}_{}_{}_{}_{}_{}'.format(poseDetector, reference_marker, noise_type, noise_magnitude, rotation_type, nRotations)
    meanName = get_mean_name(meanNameAll)
    pathMean = os.path.join(pathData_all, 'mean_{}_{}.npy'.format(partitionName, meanName))
    pathSTD = os.path.join(pathData_all, 'std_{}_{}.npy'.format(partitionName, meanName))
    if not os.path.exists(pathMean) and not os.path.exists(pathSTD): 
        print('Computing mean and standard deviation')
        features_mean_agg = np.zeros((nFeature_markers+nAddFeatures, len(idxDatasets)))
        features_std_agg = np.zeros((nFeature_markers+nAddFeatures, len(idxDatasets)))
        for c_idx, idxDataset in enumerate(idxDatasets):            
            suffix_sf = getResampleName(idxDataset, resample, num_frames)            
            c_pathDataset = os.path.join(pathData_all, '{}dataset{}_{}_{}{}{}{}'.format(prefixH5, idxDataset, num_frames, poseDetector, prefix_old_data, sensitivity_model, suffix_sf))        
            # Get partition for current dataset.    
            c_partitionName = ('{}_{}_{}'.format(augmenter_type, str(idxDataset), scaleFactorName))
            c_pathPartition = os.path.join(c_pathDataset, 'partition_{}.npy'.format(c_partitionName))
            if not os.path.exists(c_pathPartition):
                from utilities import getPartition    
                c_partition = getPartition([idxDataset], scaleFactors, infoData, subjectSplit, idxFold)
                np.save(c_pathPartition, c_partition)
            else:
                c_partition = np.load(c_pathPartition, allow_pickle=True).item()    
            c_pathMean = os.path.join(c_pathDataset, 'mean_{}_{}.npy'.format(c_partitionName, meanName))
            c_pathSTD = os.path.join(c_pathDataset, 'std_{}_{}.npy'.format(c_partitionName, meanName))
        
            if not os.path.exists(c_pathMean) and not os.path.exists(c_pathSTD):        
                print('Computing mean and standard deviation for dataset {}'.format(idxDataset))
                # Instead of accumulating data to compute mean and std, we compute them
                # on the fly so that we do not have to deal with too large matrices.
                existingAggregate = (0,0,0)    
                from utilities import update, finalize
                for count, idx in enumerate(c_partition['train']):                    
                    # We only account for 1 in 10 sequences, unecessarily long otherwise.
                    if not count % 10 == 0:
                        continue
                    # Print count every 1000.
                    if count % 1000 == 0:
                        print("{}/{}".format(count, c_partition['train'].shape[0]))                    
                    # Load time sequence.
                    # Find index sequence in current dataset.
                    idx_in_mapping = np.where(mapping['datasets'] == idxDataset)[0][0]
                    idx_adj = idx - idx_in_mapping
                    if h5:
                        with h5py.File(os.path.join(c_pathDataset, 'time_sequences.h5'), 'r') as f:
                            grp = f['data']                            
                            c_features_all = grp['features'][idx_adj, :]                       
                    else:
                        c_sequence = np.load(
                            os.path.join(c_pathDataset, "time_sequence_{}.npy".format(idx_adj)),
                            allow_pickle=True).item()
                        c_features_all = c_sequence["features"]                        
                    # Process features.
                    # Does the current dataset have arms?
                    idx_in_mapping_arms = np.where(
                        np.array(mapping['datasets_arms_idx']) == idxDataset)[0][0]
                    withArms = mapping['datasets_arms_bool'][idx_in_mapping_arms]                        
                    # 1) Extract indices features.    
                    idx_in_all_features = get_idx_in_all_features(
                        augmenter_type, poseDetector, c_features_all.shape[1], nDim=nDim, 
                        withArms=withArms, featureHeight=featureHeight, featureWeight=featureWeight)[0]
                    c_features = c_features_all[:,idx_in_all_features]
                    # 2) Express with respect to reference marker.                   
                    ref_marker_value = get_reference_marker_value(
                        c_features_all, reference_marker, poseDetector, nDim=nDim, withArms=withArms)                 
                    c_features_wrt_ref = subtract_reference_marker_value(
                        c_features, len(feature_markers), ref_marker_value, 
                        featureHeight=featureHeight, featureWeight=featureWeight)
                    # 3) Normalize by subject height.
                    height = get_height(c_features_all)
                    c_features = normalize_height(
                        c_features_wrt_ref, height, len(feature_markers), nDim=nDim, 
                        featureHeight=featureHeight, featureWeight=featureWeight)
                    if not c_features.shape[0] == num_frames:
                        raise ValueError("Dimension features is wrong")
                    # Apply rotations.
                    if withRotation:
                        for r in range(nRotations):
                            if rotation_type == 'circleRotation':
                                rotation = get_circle_rotation(nRotations, r)
                                c_features_xyz_rot = rotateArray(c_features[:,:nFeature_markers], 'y', rotation)
                            elif rotation_type == 'sphereRotation':
                                theta_x = np.arccos(2*np.random.uniform()-1)
                                theta_z = 2*np.pi*np.random.uniform() 
                                c_features_xyz_rot, _ = rotateArraySphere4(c_features[:,:nFeature_markers], ref_vec, theta_x, theta_z)
                            elif rotation_type == 'mixedCircleSphereRotation':
                                if r < nCircleRotations:                                    
                                    rotation = get_circle_rotation(nCircleRotations, r)
                                    c_features_xyz_rot = rotateArray(c_features[:,:nFeature_markers], 'y', rotation)
                                else:
                                    theta_x = np.arccos(2*np.random.uniform()-1)
                                    theta_z = 2*np.pi*np.random.uniform()
                                    c_features_xyz_rot, _ = rotateArraySphere4(c_features[:,:nFeature_markers], ref_vec, theta_x, theta_z)
                            # Add back height and weight.
                            c_features_rot = np.concatenate((c_features_xyz_rot, c_features[:,nFeature_markers:]), axis=1)
                            # Add noise.
                            if noise_magnitude > 0:
                                noise = get_noise(noise_magnitude, height, num_frames, nFeature_markers, nAddFeatures,
                                                  noise_type=noise_type, old_data=old_data)
                                c_features_rot += noise              
                            # Compute mean and std iteratively.    
                            for c_s in range(c_features_rot.shape[0]):
                                existingAggregate = update(existingAggregate, c_features_rot[c_s, :])                            
                    else:
                        # Add noise.
                        if noise_magnitude > 0:
                            noise = get_noise(noise_magnitude, height, num_frames, nFeature_markers, nAddFeatures,
                                              noise_type=noise_type, old_data=old_data)
                            c_features += noise                                            
                        # Compute mean and std iteratively.    
                        for c_s in range(c_features.shape[0]):
                            existingAggregate = update(existingAggregate, c_features[c_s, :])
                
                # Compute final mean and standard deviation.
                (features_mean_agg[:,c_idx], features_variance, _) = finalize(existingAggregate)    
                features_std_agg[:,c_idx] = np.sqrt(features_variance)
                np.save(c_pathMean, features_mean_agg[:,c_idx])
                np.save(c_pathSTD, features_std_agg[:,c_idx])

            else:
                features_mean_agg[:,c_idx] = np.load(c_pathMean)
                features_std_agg[:,c_idx] = np.load(c_pathSTD)

        features_mean_all = np.mean(features_mean_agg, axis=1)
        if len(idxDatasets) > 1:
            features_std_all = np.std(features_mean_agg, axis=1)
        else:
            features_std_all = features_std_agg.flatten()    
        np.save(pathMean, features_mean_all)
        np.save(pathSTD, features_std_all)

    else:
        print('Loading mean and standard deviation')
        features_mean_all = np.load(pathMean)
        features_std_all = np.load(pathSTD)
        
    # %% Fit all data in memory or use data generators.
    # Initialize data generators.
    # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    params = {'dim_f': (num_frames,nFeature_markers+nAddFeatures),
            'dim_r': (num_frames,nResponse_markers),
            'batch_size': batchSize,
            'shuffle': True,
            'noise_bool': noise_bool,
            'noise_type': noise_type,
            'noise_magnitude': noise_magnitude,
            'mean_subtraction': mean_subtraction,
            'std_normalization': std_normalization,
            'features_mean': features_mean_all,
            'features_std': features_std_all,
            'rotation_type': rotation_type,
            'nRotations': nRotations,
            'ref_vec': ref_vec,
            'augmenter_type': augmenter_type,
            'pose_detector': poseDetector,
            'feature_height': featureHeight,
            'feature_weight': featureWeight,
            'reference_marker': reference_marker,
            'normalize_data_height': normalizeDataHeight,
            'mapping': mapping,
            'num_frames': num_frames,
            'withRotation': withRotation,
            'sensitivity_model': sensitivity_model,                
            'h5': h5,
            'prefixH5': prefixH5,
            'old_data': old_data,
            'prefix_old_data': prefix_old_data,
            'mixedCircleSphereRotations': mixedCircleSphereRotations,
            'resample': resample
            }
    if different_data_val_loss:
        # create dict params_val_loss from params and adjust some values
        params_val_loss = params.copy()
        params_val_loss['rotation_type'] = rotation_type_val_loss
        params_val_loss['nRotations'] = nRotations_val_loss
        params_val_loss['withRotation'] = withRotation_val_loss
        params_val_loss['mixedCircleSphereRotations'] = mixedCircleSphereRotations_val_loss

    train_generator = dataGenerator(partition['train'], pathData_all, **params)
    if different_data_val_loss:
        val_generator = dataGenerator(partition_val_loss['val'], pathData_all, **params_val_loss)
    else:            
        val_generator = dataGenerator(partition['val'], pathData_all, **params)   

    # %% Initialize model.
    model = get_transformer_encoderonly_model(input_dim=nFeature_markers+nAddFeatures, output_dim=nResponse_markers,
        loss_f=loss_f, weights=weights_loss, dropout=dropout,
        num_layers=nHLayers, num_heads=nHeads, d_model=d_model, d_ff=d_ff,
        attention_axes=attention_axes, learning_r=learning_r)
           
    # Calculate total number of parameters
    # model.build(input_shape=(None, num_frames,nFeature_markers+nAddFeatures))
    # model.summary()    
    # total_params = sum([tf.reduce_prod(var.shape).numpy() for var in model.trainable_variables])
    # print(f"Total number of parameters: {total_params}")

    # %% Train model.
    if runTraining:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, verbose=1, mode="auto",
            restore_best_weights=True)
        
        # Having problems with tensorflow/cuda/smthg with training hanging
        # after the first epoch. I believe this is somehow related to the
        # validation data and the data generator. This is a hacky solution
        # to load all the validation data in memory and pass it to the
        # model.fit function.
        # Initialize empty lists to collect validation data
        X_val = []
        y_val = []
        # Iterate through the val_generator to collect data
        for batch in val_generator:
            X_batch, y_batch = batch
            X_val.append(X_batch)
            y_val.append(y_batch)
        # Convert the collected data into NumPy arrays
        X_val = np.concatenate(X_val)
        y_val = np.concatenate(y_val)
        # Use X_val and y_val instead of val_generator    
        history = model.fit(
            train_generator,  validation_data=(X_val, y_val),
            epochs=nEpochs, batch_size=batchSize, verbose=2,
            use_multiprocessing=use_multiprocessing, workers=nWorkers,
            callbacks=[early_stopping_callback])
        
    # %% Save results.
    if saveTrainedModel:
        # Save model
        augmenter = Augmenter_EncoderOnly(model)
        augmenter_instance = ExportAugmenter_EncoderOnly(augmenter)
        tf.saved_model.save(augmenter_instance, export_dir=pathTrainedModels)        
        # Save history.
        with open(os.path.join(pathTrainedModels, "history"), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)   
        # Save mean and std used for data processing.
        if mean_subtraction:
            np.save(os.path.join(pathTrainedModels, "mean.npy"), features_mean_all)
        if std_normalization:
            np.save(os.path.join(pathTrainedModels, "std.npy"), features_std_all)
        # Save partition to facilitate evaluation.
        np.save(os.path.join(pathTrainedModels, "partition.npy"), partition)
        # Add reference marker to dictionary metatdata and save as json file.
        metadata = {}
        metadata['reference_marker'] = reference_marker
        with open(os.path.join(pathTrainedModels, "metadata.json"), 'w') as fp:
            json.dump(metadata, fp)   
        
    # %% Inference testing
    if inferenceTesting:      
        # Load an input sequence
        c_pathDataset = os.path.join(pathData_all, '{}dataset{}_{}_{}{}{}{}'.format(prefixH5, idxDataset, num_frames, poseDetector, prefix_old_data, sensitivity_model, suffix_sf))
        with h5py.File(os.path.join(c_pathDataset, 'time_sequences.h5'), 'r') as f:
            grp = f['data']                            
            test_inputs_all = grp['features'][0, :]            
        idx_in_mapping_arms = np.where(
            np.array(mapping['datasets_arms_idx']) == idxDataset)[0][0]
        withArms = mapping['datasets_arms_bool'][idx_in_mapping_arms]
        idx_in_all_features = get_idx_in_all_features(
            augmenter_type, poseDetector, test_inputs_all.shape[1], nDim=nDim, 
            withArms=withArms, featureHeight=featureHeight, featureWeight=featureWeight)[0]
        test_inputs_2D = test_inputs_all[:,idx_in_all_features]
        # Get dimension output
        from utilities import get_idx_in_all_labels
        idx_labels, nResponseMarkers = get_idx_in_all_labels(
            augmenter_type, nDim=3, withArms=withArms)
        label_dimension = len(idx_labels)

        # Test running            
        augmenter_encoder_only = Augmenter_EncoderOnly(model)
        test_outputs = augmenter_encoder_only(test_inputs_2D)        
        test_outputs_np = test_outputs.numpy()
        
        # Test export
        augmenter_instance = ExportAugmenter_EncoderOnly(augmenter_encoder_only)        
        test_outputs_instance = augmenter_instance(test_inputs_2D)
        test_outputs_instance_np = test_outputs_instance.numpy()
        assert np.array_equal(test_outputs_np, test_outputs_instance_np), 'before saving'
        
        # Test saving and reload
        tf.saved_model.save(augmenter_instance, export_dir=pathTrainedModels)
        augmenter_instance_reloaded = tf.saved_model.load(pathTrainedModels)        
        test_outputs_reload = augmenter_instance_reloaded(test_inputs_2D)
        test_outputs_reload_np = test_outputs_reload.numpy()
        assert np.array_equal(test_outputs_np, test_outputs_reload_np), 'after loading'
        