'''
    ---------------------------------------------------------------------------
    utilities.py
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

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import dataman
import pandas as pd
import os
import scipy.interpolate as interpolate
import random
import string

from datasets import getInfodataset

# %% Metrics.
def getMetrics(features, responses, model,
                model_type='LSTM', encoder_only=False):
    
    if model_type == 'LSTM':
        y_pred2 = model.predict(features)
    elif model_type == 'Transformer':
        if encoder_only:            
            y_pred2 = np.zeros((responses.shape[0], responses.shape[1], responses.shape[2]))
            for i in range(responses.shape[0]):
                y_pred2[i, :, :] = model(features[i,:,:]).numpy()            
        else:
            raise ValueError('Not implemented yet.')
    elif model_type == 'linear_regression':
        y_pred2 = np.zeros((responses.shape[0], responses.shape[1], responses.shape[2]))
        for i in range(responses.shape[0]):
            y_pred2[i, :, :] = model.predict(features[i,:,:], verbose=0)            
    else:
        raise ValueError('Not implemented yet.')

    mse = np.mean(np.square(y_pred2-responses))
    rmse = np.sqrt(mse)
    return mse,rmse

def getMetrics_ind(features, responses, model):
    y_pred2 = model.predict(features)
    mse = np.mean(np.square(y_pred2-responses),axis=0)
    rmse = np.sqrt(mse)
    return mse,rmse

def getMetrics_unnorm(features, responses, model, heights):
    heights_sh = np.tile(heights, (responses.shape[1], 1)).T
    responses_unnorm = responses * heights_sh   
    y_pred2_unnorm = model.predict(features) * heights_sh
    
    mse_unnorm = np.mean(np.square(y_pred2_unnorm-responses_unnorm))
    rmse_unnorm = np.sqrt(mse_unnorm)
    return mse_unnorm,rmse_unnorm

def getMetrics_unnorm_lstm(features, responses, model, heights,
                           model_type='LSTM', encoder_only=False):
    responses_unnorm = responses * heights   

    if model_type == 'LSTM':
        y_pred2_unnorm = model.predict(features) * heights
    elif model_type == 'Transformer':
        if encoder_only:
            y_pred2 = np.zeros((responses.shape[0], responses.shape[1], responses.shape[2]))
            for i in range(responses.shape[0]):
                y_pred2[i, :, :] = model(features[i,:,:]).numpy()
            y_pred2_unnorm = y_pred2 * heights            
        else:
            raise ValueError('Not implemented yet.')
    elif model_type == 'linear_regression':
        y_pred2 = np.zeros((responses.shape[0], responses.shape[1], responses.shape[2]))
        for i in range(responses.shape[0]):
            y_pred2[i, :, :] = model.predict(features[i,:,:], verbose=0)  
        y_pred2_unnorm = y_pred2 * heights
    else:
        raise ValueError('Not implemented yet.')
    
    mse_unnorm = np.mean(np.square(y_pred2_unnorm-responses_unnorm))
    rmse_unnorm = np.sqrt(mse_unnorm)
    return mse_unnorm,rmse_unnorm
    

def getMPME_unnorm_lstm(features, responses, model, heights,
                        model_type='LSTM', encoder_only=False):
    responses_unnorm = responses * heights

    if model_type == 'LSTM':
        y_pred2_unnorm = model.predict(features) * heights
    elif model_type == 'Transformer':
        if encoder_only:
            y_pred2 = np.zeros((responses.shape[0], responses.shape[1], responses.shape[2]))
            for i in range(responses.shape[0]):
                y_pred2[i, :, :] = model(features[i,:,:]).numpy()
            y_pred2_unnorm = y_pred2 * heights
        else:
            raise ValueError('Not implemented yet.')
    elif model_type == 'linear_regression':
        y_pred2 = np.zeros((responses.shape[0], responses.shape[1], responses.shape[2]))
        for i in range(responses.shape[0]):
            y_pred2[i, :, :] = model.predict(features[i,:,:], verbose=0)
        y_pred2_unnorm = y_pred2 * heights
    else:
        raise ValueError('Not implemented yet.')
    
    # We know there are three dimensions (x,y,z).
    MPMEvec = np.zeros((int(responses_unnorm.shape[2]/3),))
    for i in range(int(responses_unnorm.shape[2]/3)):
        MPMEvec[i] = np.mean(np.linalg.norm(
            y_pred2_unnorm[:,:,i*3:i*3+3] - 
            responses_unnorm[:,:,i*3:i*3+3],axis = 2))
    MPME = np.mean(MPMEvec)
    
    return MPME, MPMEvec

def getMetrics_ind_unnorm_lstm(features, responses, model, heights,
                           model_type='LSTM', encoder_only=False):
    responses_unnorm = responses * heights       
    responses_unnorm_2D = np.reshape(responses_unnorm, 
                                      (responses_unnorm.shape[1], 
                                      responses_unnorm.shape[2])) 

    if model_type == 'LSTM':
        y_pred2_unnorm = model.predict(features) * heights
    elif model_type == 'Transformer':
        if encoder_only:
            y_pred2 = np.zeros((responses.shape[0], responses.shape[1], responses.shape[2]))
            for i in range(responses.shape[0]):
                y_pred2[i, :, :] = model(features[i,:,:]).numpy()
            y_pred2_unnorm = y_pred2 * heights     
        else:
            raise ValueError('Not implemented yet.')
    elif model_type == 'linear_regression':
        y_pred2 = np.zeros((responses.shape[0], responses.shape[1], responses.shape[2]))
        for i in range(responses.shape[0]):
            y_pred2[i, :, :] = model.predict(features[i,:,:], verbose=0)  
        y_pred2_unnorm = y_pred2 * heights
    else:
        raise ValueError('Not implemented yet.')
    
    y_pred2_unnorm_2D = np.reshape(y_pred2_unnorm, 
                                    (y_pred2_unnorm.shape[1], 
                                    y_pred2_unnorm.shape[2])) 
    
    mse = np.mean(np.square(y_pred2_unnorm_2D-responses_unnorm_2D),axis=0)
    rmse = np.sqrt(mse)
    return mse,rmse
    
def plotLossOverEpochs(history):
    plt.figure()
    plt.plot(history["loss"], linewidth=3)
    legend = ['Training']
    if 'val_loss' in history:
        plt.plot(history["val_loss"], linewidth=3)
        legend.append('Evaluation')
    plt.ylabel('loss (mean squared error)', fontsize=20)
    plt.xlabel('Epoch Number', fontsize=20)
    plt.xticks(list(range(1, len(history["loss"])+1)), fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(legend, fontsize=20)
    plt.show()
    
# Partition dataset.
def getPartition(idxDatasets, scaleFactors, infoData, subjectSplit, idxFold,
                 partial_selection={"activity":[], "factor":[]}, nScaleFactors=-1):
    
   
    idxSubject = {'train': {}, 'val': {}, 'test': {}}
    partition  = {'train': np.array([], dtype=int), 
                  'val': np.array([], dtype=int), 
                  'test': np.array([], dtype=int)}
    partition_assert  = {'train': np.array([], dtype=int), 
                         'val': np.array([], dtype=int), 
                         'test': np.array([], dtype=int)}
    count_s_train = 0
    count_s_val = 0
    count_s_test = 0
    acc_val = 0
    
    for idxDataset in idxDatasets:
        c_dataset = "dataset" + str(idxDataset)
        infoDataset = getInfodataset(idxDataset)
        
        # Select nScaleFactors out of the scaleFactors available. Different
        # selection for each dataset.
        if nScaleFactors == -1:
            c_scaleFactors = scaleFactors
        else:
            # Select nScaleFactors from scaleFactors
            c_scaleFactors = np.random.choice(scaleFactors, nScaleFactors, replace=False)
        
        for scaleFactor in c_scaleFactors:        
            # Train
            count_s = 0
            for c_idx in subjectSplit[c_dataset]["training_" + str(idxFold)]:
                c_where = np.argwhere(np.logical_and(
                    infoData["scalefactors"]==scaleFactor,
                    np.logical_and(infoData["datasets"]==idxDataset, 
                                    infoData["subjects"]==c_idx)))
                
                if infoDataset['activities'] in partial_selection['activity'] and not idxDataset in partial_selection['excluded']:
                    # Get scale factor
                    c_idx_activity = partial_selection['activity'].index(infoDataset['activities'])
                    c_scale = partial_selection['factor'][c_idx_activity]
                    # Randomly pick 1 in c_scale values from c_where
                    c_where2 = c_where[np.random.choice(
                        c_where.shape[0], int(c_where.shape[0]/c_scale), replace=False)]
                    # Sort c_where
                    c_where2 = np.sort(c_where2, axis=0)  
                else:
                    c_where2 = c_where
                
                partition["train"] = np.append(partition["train"], c_where2)
                partition_assert["train"] = np.append(partition_assert["train"], c_where)
                idxSubject["train"][count_s_train + count_s] = c_where2  
                count_s += 1
            count_s_train += count_s
            
            # Val
            count_s = 0
            for c_idx in subjectSplit[c_dataset]["validation_" + str(idxFold)]:
                c_where = np.argwhere(np.logical_and(
                    infoData["scalefactors"]==scaleFactor,
                    np.logical_and(infoData["datasets"]==idxDataset, 
                                    infoData["subjects"]==c_idx)))
                
                if infoDataset['activities'] in partial_selection['activity'] and not idxDataset in partial_selection['excluded']:
                    # Get scale factor
                    c_idx_activity = partial_selection['activity'].index(infoDataset['activities'])
                    c_scale = partial_selection['factor'][c_idx_activity]
                    # Randomly pick 1 in c_scale values from c_where
                    c_where2 = c_where[np.random.choice(
                        c_where.shape[0], int(c_where.shape[0]/c_scale), replace=False)]
                    # Sort c_where
                    c_where2 = np.sort(c_where2, axis=0)
                else:
                    c_where2 = c_where
                
                partition["val"] = np.append(partition["val"], c_where2)
                partition_assert["val"] = np.append(partition_assert["val"], c_where)
                idxSubject["val"][count_s_val + count_s] = c_where2
                count_s += 1
            count_s_val += count_s
            
            # Test
            count_s = 0
            for c_idx in subjectSplit[c_dataset]["test"]:
                c_where = np.argwhere(np.logical_and(
                    infoData["scalefactors"]==scaleFactor,
                    np.logical_and(infoData["datasets"]==idxDataset, 
                                    infoData["subjects"]==c_idx)))
                
                if infoDataset['activities'] in partial_selection['activity'] and not idxDataset in partial_selection['excluded']:
                    # Get scale factor
                    c_idx_activity = partial_selection['activity'].index(infoDataset['activities'])
                    c_scale = partial_selection['factor'][c_idx_activity]
                    # Randomly pick 1 in c_scale values from c_where
                    c_where2 = c_where[np.random.choice(
                        c_where.shape[0], int(c_where.shape[0]/c_scale), replace=False)]
                    # Sort c_where
                    c_where2 = np.sort(c_where2, axis=0)
                else:
                    c_where2 = c_where
                
                partition["test"] = np.append(partition["test"], c_where2)
                partition_assert["test"] = np.append(partition_assert["test"], c_where)
                idxSubject["test"][count_s_test + count_s] = c_where2
                count_s += 1
            count_s_test += count_s
            
            acc_val += np.sum(np.logical_and(
                infoData["datasets"] == idxDataset,
                infoData["scalefactors"]==scaleFactor))
        
    # Make sure the all the data has been split in the three sets.
    sum_partitions = partition_assert["train"].shape[0] + partition_assert["val"].shape[0] + partition_assert["test"].shape[0]
    # Relaxing test given possible split per activity
    assert (acc_val == sum_partitions), ("missing data")   
    
    return partition

# %% Welford's online algorithm: 
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)
    
# %% Markers
# Get all the markers.
def getAllMarkers():
    markers = ['Neck_mmpose', 'RShoulder_mmpose', 'LShoulder_mmpose', 'RHip_mmpose', 'LHip_mmpose',
               'midHip_mmpose', 'RKnee_mmpose', 'LKnee_mmpose', 'RAnkle_mmpose', 'LAnkle_mmpose',
               'RHeel_mmpose', 'LHeel_mmpose', 'RSmallToe_mmpose', 'LSmallToe_mmpose', 
               'RBigToe_mmpose', 'LBigToe_mmpose', 'RElbow_mmpose', 'LElbow_mmpose', 'RWrist_mmpose',
               'LWrist_mmpose', 'Neck_openpose', 'RShoulder_openpose', 'LShoulder_openpose', 
               'RHip_openpose', 'LHip_openpose', 'midHip_openpose', 'RKnee_openpose', 
               'LKnee_openpose', 'RAnkle_openpose', 'LAnkle_openpose', 'RHeel_openpose', 
               'LHeel_openpose', 'RSmallToe_openpose', 'LSmallToe_openpose', 'RBigToe_openpose',
               'LBigToe_openpose', 'RElbow_openpose', 'LElbow_openpose', 'RWrist_openpose',
               'LWrist_openpose', 'RASIS_augmenter', 'LASIS_augmenter', 'RPSIS_augmenter',
               'LPSIS_augmenter', 'RKnee_augmenter', 'RMKnee_augmenter', 'RAnkle_augmenter',
               'RMAnkle_augmenter', 'RToe_augmenter', 'R5meta_augmenter', 'RCalc_augmenter',
               'LKnee_augmenter', 'LMKnee_augmenter', 'LAnkle_augmenter', 'LMAnkle_augmenter',
               'LToe_augmenter', 'LCalc_augmenter', 'L5meta_augmenter', 'RShoulder_augmenter',
               'LShoulder_augmenter', 'C7_augmenter', 'RElbow_augmenter', 'RMElbow_augmenter',
               'RWrist_augmenter', 'RMWrist_augmenter', 'LElbow_augmenter', 'LMElbow_augmenter',
               'LWrist_augmenter', 'LMWrist_augmenter', 'RThigh1_augmenter', 'RThigh2_augmenter',
               'RThigh3_augmenter', 'LThigh1_augmenter', 'LThigh2_augmenter', 'LThigh3_augmenter',
               'RSh1_augmenter', 'RSh2_augmenter', 'RSh3_augmenter', 'LSh1_augmenter',
               'LSh2_augmenter', 'LSh3_augmenter', 'RHJC_augmenter', 'LHJC_augmenter']
    
    return markers

def getAllMarkers_oldData():
    
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe", "RElbow", "LElbow", "RWrist", "LWrist",
        "RSmallToe_mmpose", "LSmallToe_mmpose"]
    
    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]
    
    return feature_markers, response_markers

def getArmMarkersPoseDetector(pose_detector):    
    markers = ['RElbow', 'LElbow', 'RWrist', 'LWrist']
    markers = [m + "_" + pose_detector for m in markers]
    
    return markers

def getArmMarkersAugmenter():    
    markers = [ 'RElbow', 'RMElbow',
    'RWrist', 'RMWrist', 'LElbow', 'LMElbow',
    'LWrist', 'LMWrist']
    markers = [m + "_augmenter" for m in markers]
    
    return markers


def getMarkersPoseDetector(pose_detector, withArms=True):    
    markers = ['Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'midHip',
               'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RHeel', 'LHeel', 
               'RSmallToe', 'LSmallToe', 'RBigToe', 'LBigToe', 'RElbow',
               'LElbow', 'RWrist', 'LWrist']    
    markers = [m + "_" + pose_detector for m in markers]
    
    if not withArms:
        armMarkers = getArmMarkersPoseDetector(pose_detector)
        # Remove the arm markers from markers
        markers = [m for m in markers if m not in armMarkers]        
    
    return markers

def getMarkersAugmenter(withArms=True):
    markers = ['RASIS_augmenter', 'LASIS_augmenter', 'RPSIS_augmenter',
               'LPSIS_augmenter', 'RKnee_augmenter', 'RMKnee_augmenter', 'RAnkle_augmenter',
               'RMAnkle_augmenter', 'RToe_augmenter', 'R5meta_augmenter', 'RCalc_augmenter',
               'LKnee_augmenter', 'LMKnee_augmenter', 'LAnkle_augmenter', 'LMAnkle_augmenter',
               'LToe_augmenter', 'LCalc_augmenter', 'L5meta_augmenter', 'RShoulder_augmenter',
               'LShoulder_augmenter', 'C7_augmenter', 'RElbow_augmenter', 'RMElbow_augmenter',
               'RWrist_augmenter', 'RMWrist_augmenter', 'LElbow_augmenter', 'LMElbow_augmenter',
               'LWrist_augmenter', 'LMWrist_augmenter', 'RThigh1_augmenter', 'RThigh2_augmenter',
               'RThigh3_augmenter', 'LThigh1_augmenter', 'LThigh2_augmenter', 'LThigh3_augmenter',
               'RSh1_augmenter', 'RSh2_augmenter', 'RSh3_augmenter', 'LSh1_augmenter',
               'LSh2_augmenter', 'LSh3_augmenter', 'RHJC_augmenter', 'LHJC_augmenter']
    
    if not withArms:
        armMarkers = getArmMarkersAugmenter()
        # Remove the arm markers from markers
        markers = [m for m in markers if m not in armMarkers]  
    
    return markers

def getMarkersPoseDetector_lowerExtremity(pose_detector, withArms=True):
    
    markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe"]
    markers = [m + "_" + pose_detector for m in markers]
    
    allMarkers = getMarkersPoseDetector(pose_detector, withArms)
    
    idx_in_allMarkers = []
    for marker in markers:
        idx_in_allMarkers.append(allMarkers.index(marker))
        
    return markers, idx_in_allMarkers

def getOpenPoseMarkers_lowerExtremity_oldData():
    
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe"]
    
    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]
    
    all_feature_markers, all_response_markers = getAllMarkers_oldData()
    
    idx_in_all_feature_markers = []
    for marker in feature_markers:
        idx_in_all_feature_markers.append(
            all_feature_markers.index(marker))
        
    idx_in_all_response_markers = []
    for marker in response_markers:
        idx_in_all_response_markers.append(
            all_response_markers.index(marker))
        
    return feature_markers, response_markers, idx_in_all_feature_markers, idx_in_all_response_markers

def getMarkersPoseDetector_upperExtremity(pose_detector, withArms=True):
    
    markers = [
        "Neck", "RShoulder", "LShoulder", 'RElbow', 'LElbow', 
        'RWrist', 'LWrist']
    markers = [m + "_" + pose_detector for m in markers]
    
    allMarkers = getMarkersPoseDetector(pose_detector, withArms)
    
    idx_in_allMarkers = []
    for marker in markers:
        idx_in_allMarkers.append(allMarkers.index(marker))
        
    return markers, idx_in_allMarkers

def getMarkersPoseDetector_feet(pose_detector, withArms=True):
    
    markers = [
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe"]
    markers = [m + "_" + pose_detector for m in markers]
    
    allMarkers = getMarkersPoseDetector(pose_detector, withArms)
    
    idx_in_allMarkers = []
    for marker in markers:
        idx_in_allMarkers.append(allMarkers.index(marker))
        
    return markers, idx_in_allMarkers

def getMarkers_inMarkersPoseDetector(markers, pose_detector, withArms=True):
    
    allMarkers = getMarkersPoseDetector(pose_detector, withArms)
    
    idx_in_allMarkers = []
    for marker in markers:
        idx_in_allMarkers.append(allMarkers.index(marker+ "_"+ pose_detector))
        
    return idx_in_allMarkers


def getMarkersAugmenter_lowerExtremity(withArms=True):
    
    markers = [
        'RASIS_augmenter', 'LASIS_augmenter', 'RPSIS_augmenter',
        'LPSIS_augmenter', 'RKnee_augmenter', 'RMKnee_augmenter', 
        'RAnkle_augmenter', 'RMAnkle_augmenter', 'RToe_augmenter', 
        'R5meta_augmenter', 'RCalc_augmenter', 'LKnee_augmenter', 
        'LMKnee_augmenter', 'LAnkle_augmenter', 'LMAnkle_augmenter',
        'LToe_augmenter', 'LCalc_augmenter', 'L5meta_augmenter', 
        'RShoulder_augmenter', 'LShoulder_augmenter', 'C7_augmenter', 
        'RThigh1_augmenter', 'RThigh2_augmenter', 'RThigh3_augmenter',
        'LThigh1_augmenter', 'LThigh2_augmenter', 'LThigh3_augmenter',
        'RSh1_augmenter', 'RSh2_augmenter', 'RSh3_augmenter', 'LSh1_augmenter',
        'LSh2_augmenter', 'LSh3_augmenter', 'RHJC_augmenter', 'LHJC_augmenter']
    
    allMarkers = getMarkersAugmenter(withArms)
    
    idx_in_allMarkers = []
    for marker in markers:
        idx_in_allMarkers.append(allMarkers.index(marker))
        
    return markers, idx_in_allMarkers

def getMarkersAugmenter_lowerExtremity_old(withArms=True):
    
    markers = [
        "C7_augmenter", "RShoulder_augmenter", "LShoulder_augmenter",
        "RASIS_augmenter", "LASIS_augmenter", "RPSIS_augmenter", 
        "LPSIS_augmenter", "RKnee_augmenter", "LKnee_augmenter",
        "RMKnee_augmenter", "LMKnee_augmenter", "RAnkle_augmenter", 
        "LAnkle_augmenter", "RMAnkle_augmenter", "LMAnkle_augmenter",
        "RCalc_augmenter", "LCalc_augmenter", "RToe_augmenter", 
        "LToe_augmenter", "R5meta_augmenter", "L5meta_augmenter",
        "RThigh1_augmenter", "RThigh2_augmenter", "RThigh3_augmenter",
        "LThigh1_augmenter", "LThigh2_augmenter", "LThigh3_augmenter", 
        "RSh1_augmenter", "RSh2_augmenter", "RSh3_augmenter", 
        "LSh1_augmenter", "LSh2_augmenter", "LSh3_augmenter",
        "RHJC_augmenter", "LHJC_augmenter"]
    
    allMarkers = getMarkersAugmenter(withArms)
    
    idx_in_allMarkers = []
    for marker in markers:
        idx_in_allMarkers.append(allMarkers.index(marker))
        
    return markers, idx_in_allMarkers

def getMarkersAugmenter_lowerExtremityNoFeet(withArms=True):
    
    markers = [
        'RASIS_augmenter', 'LASIS_augmenter', 'RPSIS_augmenter',
        'LPSIS_augmenter', 'RKnee_augmenter', 'RMKnee_augmenter', 
        'RAnkle_augmenter', 'RMAnkle_augmenter', 'LKnee_augmenter', 
        'LMKnee_augmenter', 'LAnkle_augmenter', 'LMAnkle_augmenter',        
        'RShoulder_augmenter', 'LShoulder_augmenter', 'C7_augmenter', 
        'RThigh1_augmenter', 'RThigh2_augmenter', 'RThigh3_augmenter',
        'LThigh1_augmenter', 'LThigh2_augmenter', 'LThigh3_augmenter',
        'RSh1_augmenter', 'RSh2_augmenter', 'RSh3_augmenter', 'LSh1_augmenter',
        'LSh2_augmenter', 'LSh3_augmenter', 'RHJC_augmenter', 'LHJC_augmenter']
    
    allMarkers = getMarkersAugmenter(withArms)
    
    idx_in_allMarkers = []
    for marker in markers:
        idx_in_allMarkers.append(allMarkers.index(marker))
        
    return markers, idx_in_allMarkers

def getMarkersAugmenter_lowerExtremityNoTracking(withArms=True):
    
    markers = [
        'RASIS_augmenter', 'LASIS_augmenter', 'RPSIS_augmenter',
        'LPSIS_augmenter', 'RKnee_augmenter', 'RMKnee_augmenter', 
        'RAnkle_augmenter', 'RMAnkle_augmenter', 'RToe_augmenter', 
        'R5meta_augmenter', 'RCalc_augmenter', 'LKnee_augmenter', 
        'LMKnee_augmenter', 'LAnkle_augmenter', 'LMAnkle_augmenter',
        'LToe_augmenter', 'LCalc_augmenter', 'L5meta_augmenter', 
        'RShoulder_augmenter', 'LShoulder_augmenter', 'C7_augmenter', 
        'RHJC_augmenter', 'LHJC_augmenter']
    
    allMarkers = getMarkersAugmenter(withArms)
    
    idx_in_allMarkers = []
    for marker in markers:
        idx_in_allMarkers.append(allMarkers.index(marker))
        
    return markers, idx_in_allMarkers

def getMarkersAugmenter_lowerExtremityNoTrackingNoFeet(withArms=True):
    
    markers = [
        'RASIS_augmenter', 'LASIS_augmenter', 'RPSIS_augmenter',
        'LPSIS_augmenter', 'RKnee_augmenter', 'RMKnee_augmenter', 
        'RAnkle_augmenter', 'RMAnkle_augmenter', 'LKnee_augmenter', 
        'LMKnee_augmenter', 'LAnkle_augmenter', 'LMAnkle_augmenter',
        'RShoulder_augmenter', 'LShoulder_augmenter', 'C7_augmenter', 
        'RHJC_augmenter', 'LHJC_augmenter']
    
    allMarkers = getMarkersAugmenter(withArms)
    
    idx_in_allMarkers = []
    for marker in markers:
        idx_in_allMarkers.append(allMarkers.index(marker))
        
    return markers, idx_in_allMarkers

def getMarkersAugmenter_upperExtremity(withArms=True):
    
    markers = [
        'RElbow_augmenter', 'RMElbow_augmenter', 'RWrist_augmenter', 
        'RMWrist_augmenter', 'LElbow_augmenter', 'LMElbow_augmenter',
        'LWrist_augmenter', 'LMWrist_augmenter']
    
    allMarkers = getMarkersAugmenter(withArms)
    
    idx_in_allMarkers = []
    for marker in markers:
        idx_in_allMarkers.append(allMarkers.index(marker))
        
    return markers, idx_in_allMarkers

def getMarkersAugmenter_upperExtremity_old(withArms=True):
    
    markers = [
        "RElbow_augmenter", "LElbow_augmenter", "RMElbow_augmenter",
        "LMElbow_augmenter", "RWrist_augmenter", "LWrist_augmenter",
        "RMWrist_augmenter", "LMWrist_augmenter"]
    
    allMarkers = getMarkersAugmenter(withArms)
    
    idx_in_allMarkers = []
    for marker in markers:
        idx_in_allMarkers.append(allMarkers.index(marker))
        
    return markers, idx_in_allMarkers

def getMarkersAugmenter_feet(withArms=True):
    
    markers = [
        'RToe_augmenter', 'R5meta_augmenter', 'RCalc_augmenter', 
        'LToe_augmenter', 'L5meta_augmenter', 'LCalc_augmenter']
    
    allMarkers = getMarkersAugmenter(withArms)
    
    idx_in_allMarkers = []
    for marker in markers:
        idx_in_allMarkers.append(allMarkers.index(marker))
        
    return markers, idx_in_allMarkers

def get_idx_in_all_features(augmenter_type, poseDetector, nFeatures, nDim=3, 
                            withArms=True, featureHeight=True, 
                            featureWeight=True):
    
    if augmenter_type == 'lowerExtremity':
        markers, idx_in_allMarkers = getMarkersPoseDetector_lowerExtremity(
            poseDetector, withArms)
    elif augmenter_type == 'lowerExtremityNoTracking':
        markers, idx_in_allMarkers = getMarkersPoseDetector_lowerExtremity(
            poseDetector, withArms)
    elif augmenter_type == 'lowerExtremityNoFeet':
        markers, idx_in_allMarkers = getMarkersPoseDetector_lowerExtremity(
            poseDetector, withArms)
    elif augmenter_type == 'lowerExtremityNoTrackingNoFeet':
        markers, idx_in_allMarkers = getMarkersPoseDetector_lowerExtremity(
            poseDetector, withArms)
    elif augmenter_type == 'upperExtremity':
        markers, idx_in_allMarkers = getMarkersPoseDetector_upperExtremity(
            poseDetector, withArms)
    elif augmenter_type == 'feet':
        markers, idx_in_allMarkers = getMarkersPoseDetector_feet(
            poseDetector, withArms)
        
    # Each marker has 3 dimensions.
    idx_in_all_features = []
    for idx in idx_in_allMarkers:
        idx_in_all_features.append(idx*nDim)
        idx_in_all_features.append(idx*nDim+1)
        idx_in_all_features.append(idx*nDim+2)         
        
    # Additional features (height and weight). 
    if featureHeight:
        idxFeatureHeight = nFeatures-2
        idx_in_all_features.append(idxFeatureHeight)
    if featureWeight:
        idxFeatureWeight = nFeatures-1
        idx_in_all_features.append(idxFeatureWeight)  
        
    return idx_in_all_features, len(markers)

def get_idx_in_all_features_oldData(nDim=3, featureHeight=True, featureWeight=True):
    
    feature_markers_all, _ = getAllMarkers_oldData()    
    feature_markers, _, idx_in_all_feature_markers, _ = getOpenPoseMarkers_lowerExtremity_oldData()
    
    
    idx_in_all_features = []
    for idx in idx_in_all_feature_markers:
        idx_in_all_features.append(idx*nDim)
        idx_in_all_features.append(idx*nDim+1)
        idx_in_all_features.append(idx*nDim+2)
        
    # Additional features (height and weight). 
    nAddFeatures = 0
    if featureHeight:
        nAddFeatures += 1
        idxFeatureHeight = len(feature_markers_all)*nDim
        idx_in_all_features.append(idxFeatureHeight)
    if featureWeight:
        nAddFeatures += 1 
        if featureHeight:
            idxFeatureWeight = len(feature_markers_all)*nDim + 1
        else:
            idxFeatureWeight = len(feature_markers_all)*nDim
        idx_in_all_features.append(idxFeatureWeight)
        
    return idx_in_all_features, len(feature_markers)
    

def get_idx_in_all_labels(augmenter_type, nDim=3, withArms=True):
    
    if augmenter_type == 'lowerExtremity':
        markers, idx_in_allMarkers = getMarkersAugmenter_lowerExtremity(
            withArms)
    if augmenter_type == 'lowerExtremityNoTracking':
        markers, idx_in_allMarkers = getMarkersAugmenter_lowerExtremityNoTracking(
            withArms)
    if augmenter_type == 'lowerExtremityNoFeet':
        markers, idx_in_allMarkers = getMarkersAugmenter_lowerExtremityNoFeet(
            withArms)
    if augmenter_type == 'lowerExtremityNoTrackingNoFeet':
        markers, idx_in_allMarkers = getMarkersAugmenter_lowerExtremityNoTrackingNoFeet(
            withArms)
    elif augmenter_type == 'upperExtremity':
        markers, idx_in_allMarkers = getMarkersAugmenter_upperExtremity(
            withArms)
    elif augmenter_type == 'feet':
        markers, idx_in_allMarkers = getMarkersAugmenter_feet(
            withArms)
        
    # Each marker has 3 dimensions.
    idx_in_all_labels = []
    for idx in idx_in_allMarkers:
        idx_in_all_labels.append(idx*nDim)
        idx_in_all_labels.append(idx*nDim+1)
        idx_in_all_labels.append(idx*nDim+2)
        
    return idx_in_all_labels, len(markers)

def get_idx_in_all_labels_oldData(nDim=3):
    
    _, response_markers, _, idx_in_all_response_markers = getOpenPoseMarkers_lowerExtremity_oldData()
    
    
    idx_in_all_responses = []
    for idx in idx_in_all_response_markers:
        idx_in_all_responses.append(idx*nDim)
        idx_in_all_responses.append(idx*nDim+1)
        idx_in_all_responses.append(idx*nDim+2)
        
    return idx_in_all_responses, len(response_markers)

def get_reference_marker_value(c_features_all, reference_marker, poseDetector, 
                               nDim=3, withArms=True):
    
    # Express marker position with respect to reference marker.
    from utilities import getMarkers_inMarkersPoseDetector    
    idx_ref_in_allMarkers = getMarkers_inMarkersPoseDetector(
        [reference_marker], poseDetector, withArms)
    idx_ref_in_all_features = []
    for idx_ref_in_allMarker in idx_ref_in_allMarkers:
        idx_ref_in_all_features.append(idx_ref_in_allMarker*nDim)
        idx_ref_in_all_features.append(idx_ref_in_allMarker*nDim+1)
        idx_ref_in_all_features.append(idx_ref_in_allMarker*nDim+2)            
    c_ref = c_features_all[:,idx_ref_in_all_features]
    
    return c_ref

def subtract_reference_marker_value(c_values, nMarkers, c_ref, 
                                    featureHeight=False, 
                                    featureWeight=False):
    
    # Replicate reference marker for each marker.
    c_ref_all = np.tile(c_ref, (1, nMarkers))
    # Add columns of 0s for additional features.
    if featureHeight:
        c_ref_all = np.concatenate(
            (c_ref_all, np.zeros((c_ref_all.shape[0],1))), axis=1)
    if featureWeight:
        c_ref_all = np.concatenate(
            (c_ref_all, np.zeros((c_ref_all.shape[0],1))), axis=1)
    # Subtract reference marker from all markers.
    c_values -= c_ref_all
    
    return c_values

def get_height(c_features_all):
    
    idxFeatureHeight = c_features_all.shape[1]-2
    height = c_features_all[:,idxFeatureHeight][:,None]
    
    return height
    

def normalize_height(c_values, height, nMarkers, nDim=3, 
                     featureHeight=False, featureWeight=False):
    
    c_height_all = np.tile(height, nMarkers*nDim)
    # Add columns of 1s for additional features.
    if featureHeight:
        c_height_all = np.concatenate(
            (c_height_all, np.ones((c_height_all.shape[0],1))), axis=1)
    if featureWeight:
        c_height_all = np.concatenate(
            (c_height_all, np.ones((c_height_all.shape[0],1))), axis=1)            
    c_values /= c_height_all
    
    return c_values
    
  
# # %% Markers
# def getAllMarkers():
    
#     feature_markers = [
#         "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
#         "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
#         "RBigToe", "LBigToe", "RElbow", "LElbow", "RWrist", "LWrist",
#         "RSmallToe_mmpose", "LSmallToe_mmpose"]
    
#     response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
#                         "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
#                         "L.PSIS_study", "r_knee_study", "L_knee_study",
#                         "r_mknee_study", "L_mknee_study", "r_ankle_study", 
#                         "L_ankle_study", "r_mankle_study", "L_mankle_study",
#                         "r_calc_study", "L_calc_study", "r_toe_study", 
#                         "L_toe_study", "r_5meta_study", "L_5meta_study",
#                         "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
#                         "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
#                         "r_mwrist_study", "L_mwrist_study",
#                         "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
#                         "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
#                         "r_sh1_study", "r_sh2_study", "r_sh3_study", 
#                         "L_sh1_study", "L_sh2_study", "L_sh3_study",
#                         "RHJC_study", "LHJC_study"]
    
#     return feature_markers, response_markers

# def getOpenPoseMarkers_fullBody():
    
#     feature_markers = [
#         "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
#         "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
#         "RBigToe", "LBigToe", "RElbow", "LElbow", "RWrist", "LWrist"]
    
#     response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
#                         "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
#                         "L.PSIS_study", "r_knee_study", "L_knee_study",
#                         "r_mknee_study", "L_mknee_study", "r_ankle_study", 
#                         "L_ankle_study", "r_mankle_study", "L_mankle_study",
#                         "r_calc_study", "L_calc_study", "r_toe_study", 
#                         "L_toe_study", "r_5meta_study", "L_5meta_study",
#                         "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
#                         "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
#                         "r_mwrist_study", "L_mwrist_study",
#                         "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
#                         "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
#                         "r_sh1_study", "r_sh2_study", "r_sh3_study", 
#                         "L_sh1_study", "L_sh2_study", "L_sh3_study",
#                         "RHJC_study", "LHJC_study"]
    
#     all_feature_markers, all_response_markers = getAllMarkers()
    
#     idx_in_all_feature_markers = []
#     for marker in feature_markers:
#         idx_in_all_feature_markers.append(
#             all_feature_markers.index(marker))
        
#     idx_in_all_response_markers = []
#     for marker in response_markers:
#         idx_in_all_response_markers.append(
#             all_response_markers.index(marker))
        
#     return (feature_markers, response_markers, idx_in_all_feature_markers, 
#             idx_in_all_response_markers)

# def getMMposeMarkers_fullBody():
    
#     feature_markers = [
#         "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
#         "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe_mmpose", 
#         "LSmallToe_mmpose", "RElbow", "LElbow", "RWrist", "LWrist"]
    
#     response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
#                         "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
#                         "L.PSIS_study", "r_knee_study", "L_knee_study",
#                         "r_mknee_study", "L_mknee_study", "r_ankle_study", 
#                         "L_ankle_study", "r_mankle_study", "L_mankle_study",
#                         "r_calc_study", "L_calc_study", "r_toe_study", 
#                         "L_toe_study", "r_5meta_study", "L_5meta_study",
#                         "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
#                         "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
#                         "r_mwrist_study", "L_mwrist_study",
#                         "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
#                         "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
#                         "r_sh1_study", "r_sh2_study", "r_sh3_study", 
#                         "L_sh1_study", "L_sh2_study", "L_sh3_study",
#                         "RHJC_study", "LHJC_study"]
    
#     all_feature_markers, all_response_markers = getAllMarkers()
    
#     idx_in_all_feature_markers = []
#     for marker in feature_markers:
#         idx_in_all_feature_markers.append(
#             all_feature_markers.index(marker))
        
#     idx_in_all_response_markers = []
#     for marker in response_markers:
#         idx_in_all_response_markers.append(
#             all_response_markers.index(marker))
        
#     return (feature_markers, response_markers, idx_in_all_feature_markers, 
#             idx_in_all_response_markers)        
    
# def getOpenPoseMarkers_lowerExtremity():
    
#     feature_markers = [
#         "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
#         "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
#         "RBigToe", "LBigToe"]
    
#     response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
#                         "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
#                         "L.PSIS_study", "r_knee_study", "L_knee_study",
#                         "r_mknee_study", "L_mknee_study", "r_ankle_study", 
#                         "L_ankle_study", "r_mankle_study", "L_mankle_study",
#                         "r_calc_study", "L_calc_study", "r_toe_study", 
#                         "L_toe_study", "r_5meta_study", "L_5meta_study",
#                         "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
#                         "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
#                         "r_sh1_study", "r_sh2_study", "r_sh3_study", 
#                         "L_sh1_study", "L_sh2_study", "L_sh3_study",
#                         "RHJC_study", "LHJC_study"]
    
#     all_feature_markers, all_response_markers = getAllMarkers()
    
#     idx_in_all_feature_markers = []
#     for marker in feature_markers:
#         idx_in_all_feature_markers.append(
#             all_feature_markers.index(marker))
        
#     idx_in_all_response_markers = []
#     for marker in response_markers:
#         idx_in_all_response_markers.append(
#             all_response_markers.index(marker))
        
#     return (feature_markers, response_markers, idx_in_all_feature_markers, 
#             idx_in_all_response_markers)

# def getMMposeMarkers_lowerExtremity():
    
#     feature_markers = [
#         "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
#         "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe_mmpose", 
#         "LSmallToe_mmpose"]
    
#     response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
#                         "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
#                         "L.PSIS_study", "r_knee_study", "L_knee_study",
#                         "r_mknee_study", "L_mknee_study", "r_ankle_study", 
#                         "L_ankle_study", "r_mankle_study", "L_mankle_study",
#                         "r_calc_study", "L_calc_study", "r_toe_study", 
#                         "L_toe_study", "r_5meta_study", "L_5meta_study",
#                         "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
#                         "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
#                         "r_sh1_study", "r_sh2_study", "r_sh3_study", 
#                         "L_sh1_study", "L_sh2_study", "L_sh3_study",
#                         "RHJC_study", "LHJC_study"]
    
#     all_feature_markers, all_response_markers = getAllMarkers()
    
#     idx_in_all_feature_markers = []
#     for marker in feature_markers:
#         idx_in_all_feature_markers.append(
#             all_feature_markers.index(marker))
        
#     idx_in_all_response_markers = []
#     for marker in response_markers:
#         idx_in_all_response_markers.append(
#             all_response_markers.index(marker))
        
#     return (feature_markers, response_markers, idx_in_all_feature_markers, 
#             idx_in_all_response_markers)
    
# def getMarkers_upperExtremity_pelvis():
    
#     feature_markers = [
#         "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RElbow", "LElbow",
#         "RWrist", "LWrist"]
    
#     response_markers = ["r_lelbow_study", "L_lelbow_study", "r_melbow_study",
#                         "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
#                         "r_mwrist_study", "L_mwrist_study"]
    
#     all_feature_markers, all_response_markers = getAllMarkers()
    
#     idx_in_all_feature_markers = []
#     for marker in feature_markers:
#         idx_in_all_feature_markers.append(
#             all_feature_markers.index(marker))
        
#     idx_in_all_response_markers = []
#     for marker in response_markers:
#         idx_in_all_response_markers.append(
#             all_response_markers.index(marker))
        
#     return (feature_markers, response_markers, idx_in_all_feature_markers,
#             idx_in_all_response_markers)

# def getMarkers_upperExtremity_noPelvis():
    
#     feature_markers = [
#         "Neck", "RShoulder", "LShoulder", "RElbow", "LElbow", 
#         "RWrist", "LWrist"]
    
#     response_markers = ["r_lelbow_study", "L_lelbow_study", "r_melbow_study",
#                         "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
#                         "r_mwrist_study", "L_mwrist_study"]
    
#     all_feature_markers, all_response_markers = getAllMarkers()
    
#     idx_in_all_feature_markers = []
#     for marker in feature_markers:
#         idx_in_all_feature_markers.append(
#             all_feature_markers.index(marker))
        
#     idx_in_all_response_markers = []
#     for marker in response_markers:
#         idx_in_all_response_markers.append(
#             all_response_markers.index(marker))
        
#     return (feature_markers, response_markers, idx_in_all_feature_markers, 
#             idx_in_all_response_markers)

# %% Rotate data.
def rotateArray(data, axis, value, inDegrees=True):
    
    assert np.mod(data.shape[1],3) == 0, 'wrong dimension rotateArray'
    r = R.from_euler(axis, value, degrees=inDegrees)     
    
    data_out = np.zeros((data.shape[0], data.shape[1]))
    for i in range(int(data.shape[1]/3)):
        c = data[:,i*3:(i+1)*3]        
        data_out[:,i*3:(i+1)*3] = r.apply(c)
        
    return data_out

# %% Rotate data (sphere).
# Based on first approach from:
# http://paulbourke.net/geometry/randomvector/
def rotateArraySphere1(data, theta_x, theta_y, theta_z, inDegrees=False):    
    assert np.mod(data.shape[1],3) == 0, 'wrong dimension rotateArray'
    
    r1 = R.from_euler('x', theta_x, degrees=False)
    r2 = R.from_euler('y', theta_z, degrees=False)
    r3 = R.from_euler('z', theta_z, degrees=False)
    
    data_out = np.zeros((data.shape[0], data.shape[1]))
    for i in range(int(data.shape[1]/3)):
        c = data[:,i*3:(i+1)*3]        
        unit_vec_r1 = r1.apply(c)
        unit_vec_r2 = r2.apply(unit_vec_r1)
        data_out[:,i*3:(i+1)*3] = r3.apply(unit_vec_r2)
        
    return data_out

# %% Rotate data (sphere).
# Based on second approach from:
# http://paulbourke.net/geometry/randomvector/
def rotateArraySphere2(data, theta_x, theta_z, inDegrees=False):    
    assert np.mod(data.shape[1],3) == 0, 'wrong dimension rotateArray'
    
    r1 = R.from_euler('x', theta_x, degrees=inDegrees)
    r2 = R.from_euler('z', theta_z, degrees=inDegrees)
    
    data_out = np.zeros((data.shape[0], data.shape[1]))
    for i in range(int(data.shape[1]/3)):
        c = data[:,i*3:(i+1)*3]
        unit_vec_r1 = r1.apply(c)
        data_out[:,i*3:(i+1)*3] = r2.apply(unit_vec_r1)
        
    return data_out

# %% Rotate data (sphere).
# Based on third approach from:
# http://paulbourke.net/geometry/randomvector/
def get_rotation_matrix(vec2, vec1=np.array([1, 0, 0])):
    """get rotation matrix between two vectors using scipy"""
    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))
    r = R.align_vectors(vec2, vec1)
    return r[0]

def rotateArraySphere3(data, ref_vec, theta_x, theta_z, unit_vec=np.array([0,0,0]),
                       inDegrees=False):    
    assert np.mod(data.shape[1],3) == 0, 'wrong dimension '
    
    r1 = R.from_euler('x', theta_x, degrees=inDegrees)
    r2 = R.from_euler('z', theta_z, degrees=inDegrees)
    
    if not np.any(unit_vec):
        unit_vec = data[0, 0:3]
    r_align = get_rotation_matrix(vec1=unit_vec, vec2=ref_vec)
    
    data_out = np.zeros((data.shape[0], data.shape[1]))
    for i in range(int(data.shape[1]/3)):
        for j in range(data.shape[0]):
            c = data[j,i*3:(i+1)*3]        
            unit_vec_align = r_align.apply(c)            
            unit_vec_r1 = r1.apply(unit_vec_align)
            data_out[j,i*3:(i+1)*3] = r2.apply(unit_vec_r1)
        
    return data_out, unit_vec

# Faster version of rotateArraySphere3
def rotateArraySphere4(data, ref_vec, theta_x, theta_z, unit_vec=np.array([0,0,0]),
                       inDegrees=False):    
    assert np.mod(data.shape[1],3) == 0, 'wrong dimension rotateArray'
    
    r1 = R.from_euler('x', theta_x, degrees=inDegrees)
    r2 = R.from_euler('z', theta_z, degrees=inDegrees)
    
    if not np.any(unit_vec):
        unit_vec = data[0, :3]
    
    r_align = get_rotation_matrix(vec1=unit_vec, vec2=ref_vec)
    data_out = np.zeros((data.shape[0], data.shape[1]))
    for i in range(int(data.shape[1]/3)):    
        unit_vec_align = r_align.apply(data[:, i*3:(i+1)*3])
        unit_vec_r1 = r1.apply(unit_vec_align)
        data_out[:,i*3:(i+1)*3] = r2.apply(unit_vec_r1)
        
    return data_out, unit_vec

# %% TRC format to numpy format.
def TRC2numpy(pathFile, markers):
    
    trc_file = dataman.TRCFile(pathFile)
    time = trc_file.time
    num_frames = time.shape[0]
    data = np.zeros((num_frames, len(markers)*3))
    for count, marker in enumerate(markers):
        data[:,3*count:3*count+3] = trc_file.marker(marker)    
    this_dat = np.empty((num_frames, 1))
    this_dat[:, 0] = time
    data_out = np.concatenate((this_dat, data), axis=1)
    
    return data_out

# %%  Found here: https://github.com/chrisdembia/perimysium/
def storage2numpy(storage_file, excess_header_entries=0):
    """Returns the data from a storage file in a numpy format. Skips all lines
    up to and including the line that says 'endheader'.
    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.
    Returns
    -------
    data : np.ndarray (or numpy structure array or something?)
        Contains all columns from the storage file, indexable by column name.
    excess_header_entries : int, optional
        If the header row has more names in it than there are data columns.
        We'll ignore this many header row entries from the end of the header
        row. This argument allows for a hacky fix to an issue that arises from
        Static Optimization '.sto' outputs.
    Examples
    --------
    Columns from the storage file can be obtained as follows:
        >>> data = storage2numpy('<filename>')
        >>> data['ground_force_vy']
    """
    # What's the line number of the line containing 'endheader'?
    f = open(storage_file, 'r')

    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count('endheader') != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()

    # With this information, go get the data.
    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1
    data = np.genfromtxt(storage_file, names=names,
            skip_header=skip_header)

    return data

def storage2df(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])    
    
    return out

def numpy2TRC(f, data, headers, fc=50.0, t_start=0.0, units="m"):
    
    header_mapping = {}
    for count, header in enumerate(headers):
        header_mapping[count+1] = header 
        
    # Line 1.
    f.write('PathFileType  4\t(X/Y/Z) %s\n' % os.getcwd())
    
    # Line 2.
    f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\t'
                'Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
    
    num_frames=data.shape[0]
    num_markers=len(header_mapping.keys())
    
    # Line 3.
    f.write('%.1f\t%.1f\t%i\t%i\t%s\t%.1f\t%i\t%i\n' % (
            fc, fc, num_frames,
            num_markers, units, fc,
            1, num_frames))
    
    # Line 4.
    f.write("Frame#\tTime\t")
    for key in sorted(header_mapping.keys()):
        f.write("%s\t\t\t" % format(header_mapping[key]))

    # Line 5.
    f.write("\n\t\t")
    for imark in np.arange(num_markers) + 1:
        f.write('X%i\tY%s\tZ%s\t' % (imark, imark, imark))
    f.write('\n')
    
    # Line 6.
    f.write('\n')

    for frame in range(data.shape[0]):
        f.write("{}\t{:.8f}\t".format(frame,(frame)/fc+t_start))

        for key in sorted(header_mapping.keys()):
            f.write("{:.8f}\t{:.8f}\t{:.8f}\t".format(data[frame,0+(key-1)*3], data[frame,1+(key-1)*3], data[frame,2+(key-1)*3]))
        f.write("\n")


# Create OpenSim function that returns the state trajectory given paths to a model and motion file.
def get_state_trajectory(model_file_path, motion_file_path):
    import opensim
    # Load the model and motion files.
    model = opensim.Model(model_file_path)
    model.initSystem()

    # Create time-series table with coordinate values.             
    table = opensim.TimeSeriesTable(motion_file_path)        
    tableProcessor = opensim.TableProcessor(table)
    tableProcessor.append(opensim.TabOpUseAbsoluteStateNames())
    time = np.asarray(table.getIndependentColumn())

    # Convert in radians.
    table = tableProcessor.processAndConvertToRadians(model)
                
    # Compute coordinate speeds and add to table.        
    Qs = table.getMatrix().to_numpy()
    Qds = np.zeros(Qs.shape)
    columnAbsoluteLabels = list(table.getColumnLabels())
    for i, columnLabel in enumerate(columnAbsoluteLabels):
        spline = interpolate.InterpolatedUnivariateSpline(
            time, Qs[:,i], k=3)
        # Coordinate speeds
        splineD1 = spline.derivative(n=1)
        Qds[:,i] = splineD1(time)          
        # Add coordinate speeds to table.
        columnLabel_speed = columnLabel[:-5] + 'speed'
        table.appendColumn(
            columnLabel_speed, 
            opensim.Vector(Qds[:,i].flatten().tolist()))
            
    # Append missing muscle states to table, set 0s to everything
    # Needed for StatesTrajectory.
    stateVariableNames = model.getStateVariableNames()
    stateVariableNamesStr = [
        stateVariableNames.get(i) for i in range(
            stateVariableNames.getSize())]
    existingLabels = table.getColumnLabels()
    for stateVariableNameStr in stateVariableNamesStr:
        if not stateVariableNameStr in existingLabels:
            vec_0 = opensim.Vector([0] * table.getNumRows())            
            table.appendColumn(stateVariableNameStr, vec_0)
            
    # Set state trajectory
    stateTrajectory = opensim.StatesTrajectory.createFromStatesTable(
        model, table)

    return table, stateTrajectory  

def get_circle_rotation(nRotations, r):
    
    rotations = [i*360/nRotations for i in range(0,nRotations)] + [360.]
    rotation = np.random.choice(np.arange(rotations[r], rotations[r+1], 1), size=1)[0]
    
    return rotation

def get_noise(noise_magnitude, height, num_frames, nFeature_markers, nAddFeatures,
              noise_type="per_timestep", old_data=False):
    
    # Normalize noise magnitude by subject height if not old data.
    if old_data:
        noise_magnitude_norm = noise_magnitude
    else:
        noise_magnitude_norm = noise_magnitude / height[0][0]                        
    if noise_type == "per_timestep":
        noise = np.zeros((num_frames, nFeature_markers+nAddFeatures))
        noise[:,:nFeature_markers] = np.random.normal(
            0, noise_magnitude_norm, (num_frames, nFeature_markers))                        
    else:
        raise ValueError("Only per_timestep noise type supported")
        
    return noise
    

# %%  Numpy array to storage file.
def numpy_to_storage(labels, data, storage_file, datatype=None):
    
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"
    
    f = open(storage_file, 'w')
    # Old style
    if datatype is None:
        f = open(storage_file, 'w')
        f.write('name %s\n' %storage_file)
        f.write('datacolumns %d\n' %data.shape[1])
        f.write('datarows %d\n' %data.shape[0])
        f.write('range %f %f\n' %(np.min(data[:, 0]), np.max(data[:, 0])))
        f.write('endheader \n')
    # New style
    else:
        if datatype == 'IK':
            f.write('Coordinates\n')
        elif datatype == 'ID':
            f.write('Inverse Dynamics Generalized Forces\n')
        elif datatype == 'GRF':
            f.write('%s\n' %storage_file)
        elif datatype == 'muscle_forces':
            f.write('ModelForces\n')
        f.write('version=1\n')
        f.write('nRows=%d\n' %data.shape[0])
        f.write('nColumns=%d\n' %data.shape[1])    
        if datatype == 'IK':
            f.write('inDegrees=yes\n\n')
            f.write('Units are S.I. units (second, meters, Newtons, ...)\n')
            f.write("If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\n")
        elif datatype == 'ID':
            f.write('inDegrees=no\n')
        elif datatype == 'GRF':
            f.write('inDegrees=yes\n')
        elif datatype == 'muscle_forces':
            f.write('inDegrees=yes\n\n')
            f.write('This file contains the forces exerted on a model during a simulation.\n\n')
            f.write("A force is a generalized force, meaning that it can be either a force (N) or a torque (Nm).\n\n")
            f.write('Units are S.I. units (second, meters, Newtons, ...)\n')
            f.write('Angles are in degrees.\n\n')
            
        f.write('endheader \n')
    
    for i in range(len(labels)):
        f.write('%s\t' %labels[i])
    f.write('\n')
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write('%20.8f\t' %data[i, j])
        f.write('\n')
        
    f.close()
    
def get_partition_name(partitionNameAll):

    if partitionNameAll == 'lowerExtremity_012345678910111213141516171819202122232425262728_10_-1_walking9_acl8_standing15_cycling8_karate8_tennis2':
        random.seed(1)
    elif partitionNameAll == 'lowerExtremity_012345678910111213141516171819202122232425262728_090951010511_-1_walking9_acl8_standing15_cycling8_karate8_tennis2':
        random.seed(2)
    elif partitionNameAll == 'lowerExtremity_012345678910111213141516171819202122232425262728_090951010511_1_walking9_acl8_standing15_cycling8_karate8_tennis2':
        random.seed(3)
    elif partitionNameAll == 'lowerExtremity_012345678910111213141516171819202122232425262728_090951010511_2_walking9_acl8_standing15_cycling8_karate8_tennis2':
        random.seed(4)
    elif partitionNameAll == 'lowerExtremity_012345678910111213141516171819202122232425262728_090951010511_3_walking9_acl8_standing15_cycling8_karate8_tennis2':
        random.seed(0)
    # Duplicated because we added the distinction between walking and treadmill_walking afterwards
    elif partitionNameAll == 'lowerExtremity_012345678910111213141516171819202122232425262728_090951010511_3_walking9_treadmillwalking9_acl8_standing15_cycling8_karate8_tennis2':
        random.seed(0)
    elif partitionNameAll == 'lowerExtremity_012345678910111213141516171819202122232425262728_090951010511_4_walking9_acl8_standing15_cycling8_karate8_tennis2':
        random.seed(5)
    elif partitionNameAll == 'lowerExtremity_012345678910111213141516171819202122232425262728_090951010511_5_walking9_acl8_standing15_cycling8_karate8_tennis2':
        random.seed(6)
    elif partitionNameAll == 'upperExtremity_0_090951010511_1_walking6_acl4_standing15_cycling8_karate8':
        random.seed(7)
    elif partitionNameAll == 'upperExtremity_0345681011181920212223242728_090951010511_3_walking6_acl4_standing15_cycling8_karate8':
        random.seed(8)   
    # Duplicated because we added the distinction between walking and treadmill_walking afterwards
    elif partitionNameAll == 'upperExtremity_0345681011181920212223242728_090951010511_3_walking6_treadmillwalking6_acl4_standing15_cycling8_karate8':
        random.seed(8)  
    elif partitionNameAll == 'upperExtremity_3_10_-1_walking6_acl4_standing15_cycling8_karate8':
        random.seed(9)
    elif partitionNameAll == 'lowerExtremity_012345678910111213141516171819202122232425262728_090951010511_-1_walking1_treadmillwalking13_acl8_standing15_cycling8_karate8_tennis2':
        random.seed(10)
    elif partitionNameAll == 'upperExtremity_0345681011181920212223242728_090951010511_-1_walking1_treadmillwalking6_acl4_standing15_cycling8_karate8':
        random.seed(11)
    elif partitionNameAll == 'upperExtremity_3_10_-1_walking1_treadmillwalking6_acl4_standing15_cycling8_karate8':
        random.seed(12)
    elif partitionNameAll == 'feet_012345678910111213141516171819202122232425262728_090951010511_-1_walking1_treadmillwalking13_acl8_standing15_cycling8_karate8_tennis2':
        random.seed(13)
    elif partitionNameAll == 'lowerExtremity_1234567812242526_090951010511_-1_walking1_treadmillwalking13_standing15':
        random.seed(14)
    elif partitionNameAll == 'lowerExtremityNoTracking_1234567812242526_090951010511_-1_walking1_treadmillwalking13_standing15':
        random.seed(15)
    elif partitionNameAll == 'feet_1_090951010511_-1':
        random.seed(16)
    elif partitionNameAll == 'lowerExtremityNoTrackingNoFeet_1_090951010511_-1':
        random.seed(17)
    elif partitionNameAll == 'lowerExtremityNoTrackingNoFeet_12345678242526_090951010511_-1_treadmillwalking15':
        random.seed(18)
    elif partitionNameAll == 'lowerExtremityNoTracking_12345678242526_090951010511_-1_treadmillwalking15':
        random.seed(19)
    elif partitionNameAll == 'feet_12345678242526_090951010511_-1_treadmillwalking15':
        random.seed(20)
    elif partitionNameAll == 'lowerExtremityNoTracking_1_090951010511_-1':
        random.seed(21)
    elif partitionNameAll == 'lowerExtremity_12345678242526_090951010511_-1_treadmillwalking15':
        random.seed(22)
    elif partitionNameAll == 'lowerExtremity_12345678242526_090951010511_-1_treadmillwalking5':
        random.seed(23)
    elif partitionNameAll == 'lowerExtremity_12345678242526_090951010511_-1':
        random.seed(24)
    elif partitionNameAll == 'lowerExtremity_0_090951010511_-1':
        random.seed(25)
    elif partitionNameAll == 'lowerExtremity_12459_090951010511_-1':
        random.seed(26)
    elif partitionNameAll == 'lowerExtremity_4_09_-1':
        random.seed(26)
    elif partitionNameAll == 'lowerExtremity_3_090951010511_-1':
        random.seed(27)
    elif partitionNameAll == 'lowerExtremity_0123456789_090951010511_-1':
        random.seed(28)
    elif partitionNameAll == 'lowerExtremity_4_10_-1':
        random.seed(29)
    elif partitionNameAll == 'lowerExtremity_0123456789_090951010511_-1_treadmillwalking15':
        random.seed(30)
    elif partitionNameAll == 'lowerExtremity_0_090951010511_-1':
        random.seed(31)
    # Starting to work w/ new data re-sampled at 60Hz.
    elif partitionNameAll == 'lowerExtremity_012345678910111213141516171819202122232425262728_090951010511_-1_treadmillwalking5_acl2_standing20_cycling3_karate2':
        random.seed(100)
    # Used non curated excluded subjects
    # elif partitionNameAll == 'lowerExtremity_0125789101113141516171819202122232425262728_090951010511_-1_treadmillwalking4_cycling3_karate2':
    #     random.seed(101)
    # Used curated excluded subjects
    elif partitionNameAll == 'lowerExtremity_0125789101113141516171819202122232425262728_090951010511_-1_treadmillwalking4_cycling3_karate2':
        random.seed(102)
    elif partitionNameAll == 'lowerExtremityNoFeet_0125789101113141516171819202122232425262728_090951010511_-1_treadmillwalking4_cycling3_karate2':
        random.seed(103)
    elif partitionNameAll == 'feet_0125789101113141516171819202122232425262728_090951010511_-1_treadmillwalking4_cycling3_karate2':
        random.seed(104)
    elif partitionNameAll == 'upperExtremity_0581011181920212223242728_090951010511_-1_cycling2':
        random.seed(105)
    elif partitionNameAll == 'lowerExtremity_0125789101113141516171819202122232425262728_090951010511_-1':
        random.seed(106)
    elif partitionNameAll == 'lowerExtremity_0125789101113141516171819202122232425262728_090951010511_-1_treadmillwalking4_cycling3_karate2_full2':
        random.seed(107)
    elif partitionNameAll == 'lowerExtremity_0125789101113141516171819202122232425262728_090951010511_-1_treadmillwalking2_cycling3_karate2_full2':
        random.seed(108)
    elif partitionNameAll == 'lowerExtremity_0_090951010511_-1':
        random.seed(109)
    elif partitionNameAll == 'upperExtremity_0581011181920212223242728_090951010511_-1_cycling2_full2':
        random.seed(110)
    elif partitionNameAll == 'lowerExtremity_0125789101113141516171819202122232425262728_090951010511_-1_cycling3_karate2':
        random.seed(111)
    elif partitionNameAll == 'lowerExtremity_0_10_-1':
        random.seed(112)
    elif partitionNameAll == 'upperExtremity_0_090951010511_-1':
        random.seed(113)
    elif partitionNameAll == 'lowerExtremity_0125789101113141516171819202122232425262728_10_-1_treadmillwalking2_cycling3_karate2_full2':
        random.seed(114)
    elif partitionNameAll == 'upperExtremity_0581011181920212223242728_10_-1_cycling2':
        random.seed(115)
    elif partitionNameAll == 'lowerExtremity_01257891011131415161718192021222324252627_090951010511_-1_treadmillwalking2_cycling3_karate2_full2':
        random.seed(116)
    elif partitionNameAll == 'upperExtremity_05810111819202122232427_090951010511_-1_cycling2_full2':
        random.seed(117)
    elif partitionNameAll == 'lowerExtremity_01257891011131415161718192021222324252627_10_-1_treadmillwalking2_cycling3_karate2_full2':
        random.seed(118)
    elif partitionNameAll == 'upperExtremity_05810111819202122232427_090951010511_-1_cycling2':
        random.seed(119)
    elif partitionNameAll == 'upperExtremity_05810111819202122232427_10_-1_cycling2':
        random.seed(119)
    # Test
    elif partitionNameAll == 'lowerExtremity_4_09_-1':
        random.seed(1000)

    # Demo
    elif partitionNameAll == 'lowerExtremity_0_10_-1':
        random.seed(10000)
    elif partitionNameAll == 'upperExtremity_0_10_-1':
        random.seed(10001)

    else:
        raise ValueError("TODO: add name partition: {}".format(partitionNameAll))
    partitionName = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        
    return partitionName

def get_mean_name(meanNameAll):
    
    if meanNameAll == 'openpose_midHip_per_timestep_0.018_sphereRotation_1':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.018_sphereRotation_1':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.018_sphereRotation_1':
        random.seed(2)
    # Use same mean/std as for openpose_midHip_per_timestep_0.018_sphereRotation_1
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_sphereRotation_2':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.018_sphereRotation_2':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.018_sphereRotation_2':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_sphereRotation_3':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.018_sphereRotation_3':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.018_sphereRotation_3':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_sphereRotation_4':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.018_sphereRotation_4':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.018_sphereRotation_4':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_sphereRotation_5':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.018_sphereRotation_5':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.018_sphereRotation_5':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_sphereRotation_6':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.018_sphereRotation_6':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.018_sphereRotation_6':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_circleRotation_1':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.018_circleRotation_1':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.018_circleRotation_1':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_circleRotation_2':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.018_circleRotation_2':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.018_circleRotation_2':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_circleRotation_3':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.018_circleRotation_3':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.018_circleRotation_3':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_circleRotation_4':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.018_circleRotation_4':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.018_circleRotation_4':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_circleRotation_5':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.018_circleRotation_5':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.018_circleRotation_5':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_circleRotation_6':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.018_circleRotation_6':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.018_circleRotation_6':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_circleRotation_10':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.018_circleRotation_10':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.018_circleRotation_10':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.009_circleRotation_1':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.009_circleRotation_1':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.009_circleRotation_1':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.009_circleRotation_9':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.009_circleRotation_9':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.009_circleRotation_9':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.009_circleRotation_10':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.009_circleRotation_10':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.009_circleRotation_10':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_circleRotation_9':
        random.seed(0)
    elif meanNameAll == 'openpose_RKnee_per_timestep_0.018_circleRotation_9':
        random.seed(1)
    elif meanNameAll == 'openpose_Neck_per_timestep_0.018_circleRotation_9':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_circleRotation_1':
        random.seed(0)        
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_mixedCircleSphereRotation_5':
        random.seed(100)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_mixedCircleSphereRotation_10':
        random.seed(101)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_mixedCircleSphereRotation_7':
        random.seed(101)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_mixedCircleSphereRotation_12':
        random.seed(101)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_mixedCircleSphereRotation_8':
        random.seed(101)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_mixedCircleSphereRotation_2':
        random.seed(2)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.009_mixedCircleSphereRotation_7':
        random.seed(101)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.009_mixedCircleSphereRotation_12':
        random.seed(101)
    elif meanNameAll == 'openpose_midHip_per_timestep_0.018_mixedCircleSphereRotation_2':
        random.seed(1000)

    elif meanNameAll == 'mmpose_midHip_per_timestep_0.018_mixedCircleSphereRotation_8':
        random.seed(102)
    else:
        raise ValueError("TODO: add name mean")
    meanName = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        
    return meanName

def getInfoDataName(infoDataNameAll):

    if infoDataNameAll == '2-100fs':
        random.seed(0)
    elif infoDataNameAll == '2-200fs':
        random.seed(0)
    else:
        raise ValueError("TODO: add info data name")
    infoDataName = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

    return infoDataName

def getResampleName(idxDataset, resample, num_frames):
    
    if int(idxDataset) in resample["Dataset"]:
        fs = resample["fs"][resample["Dataset"].index(int(idxDataset))]
        if num_frames == 60:
            if not fs == 60:
                suffix_sf = "_fs" + str(fs)
            else:
                suffix_sf = ""
        elif num_frames == 30:
            if not fs == 60:
                suffix_sf = "_fs" + str(fs)
            else:
                suffix_sf = ""
        elif num_frames == 120:
            if not fs == 120:
                suffix_sf = "_fs" + str(fs)
            else:
                suffix_sf = ""
        else:
            suffix_sf = ""
    else:
        suffix_sf = ""
        
    return suffix_sf
