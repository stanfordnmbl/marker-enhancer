import numpy as np
from tensorflow import keras
import os
from utilities import rotateArray, rotateArraySphere4, get_idx_in_all_features, get_idx_in_all_labels
from utilities import get_reference_marker_value, subtract_reference_marker_value, get_circle_rotation
from utilities import get_height, normalize_height, get_idx_in_all_features_oldData, get_idx_in_all_labels_oldData, getResampleName
import h5py

# Inspired from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class dataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, pathData, batch_size=64, dim_f=(30,59), 
                 dim_r=(30,87), shuffle=True, noise_bool=False, noise_type='',
                 noise_magnitude=0, mean_subtraction=False, 
                 std_normalization=False, features_mean=0, features_std=0,
                 rotation_type='circleRotation', nRotations=0, 
                 ref_vec=np.array([0,0,1]), augmenter_type='lowerExtremity',
                 pose_detector='openpose', feature_height=True, 
                 feature_weight=True, reference_marker='midHip',
                 normalize_data_height=True, mapping={}, num_frames=60,
                 withRotation=False, sensitivity_model='', h5=False,
                 prefixH5='', old_data=False, prefix_old_data='',
                 mixedCircleSphereRotations={}, resample={}):
        'Initialization'
        self.dim_f = dim_f
        self.dim_r = dim_r
        
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        
        self.shuffle = shuffle
        self.on_epoch_end()
        
        self.pathData = pathData
        
        self.noise_bool = noise_bool
        self.noise_magnitude = noise_magnitude 
        
        self.mean_subtraction = mean_subtraction
        self.std_normalization = std_normalization
        self.features_mean = features_mean
        self.features_std = features_std
        
        self.augmenter_type = augmenter_type
        self.pose_detector = pose_detector
        self.sensitivity_model = sensitivity_model
        
        self.h5 = h5
        self.prefixH5 = prefixH5
        
        self.old_data = old_data
        self.prefix_old_data = prefix_old_data
        
        self.feature_height = feature_height
        self.feature_weight = feature_weight
        self.reference_marker = reference_marker
        self.normalize_data_height = normalize_data_height
        
        self.mapping = mapping     
        self.num_frames = num_frames
        
        self.withRotation = withRotation
        self.rotation_type = rotation_type
        self.nRotations = nRotations
        self.ref_vec = ref_vec        
        if not self.withRotation:            
            self.nRotations = 1 # Set to 1 to get proper indexes.
        self.mixedCircleSphereRotations = mixedCircleSphereRotations
        
        self.resample = resample

    def __len__(self):
        'Denotes the number of batches per epoch'        
        # We multiply the number of samples by the number of rotations we want
        # to apply to each sample, and then divide by the batch size.
        return int(np.floor(
            (len(self.list_IDs)*(self.nRotations)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch        
        # Take into account the rotation: we have nRotations x the same sample.
        # Eg,   list_IDs = [0,1,2,3,4,5,6,7,8,9,10,11]
        #       batch_size = 3, n_rot = 4, n_batch = (12*4)/3=16
        #       1st 4 indices correspond to list_IDs0, rot0-4
        #       next 4 indices correspond to list_IDs1, rot0-4
        #       ...
        #       index => floor(index/n_rot)
        #           idx0=0, idx1=0, idx2=0, idx3=0, idx4=1, idx5=1, idx6=1, ...
        index_rot = int(np.floor(index/(self.nRotations)))
        indexes = self.indexes[index_rot*self.batch_size:(index_rot+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]        

        #       ...
        #       index => mod(index, n_rot)
        #           idx0=0, idx1=1, idx2=2, idx3=3, idx4=0, idx5=1, idx6=2, ...
        idx_rot = np.mod(index, (self.nRotations))

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, idx_rot)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, idx_rot):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim_f))
        y = np.empty((self.batch_size, *self.dim_r))
                
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Pick a rotation.
            if self.withRotation:
                if self.rotation_type == 'circleRotation':
                    rotation = get_circle_rotation(self.nRotations, idx_rot)
                elif self.rotation_type == 'sphereRotation':        
                    theta_x = np.arccos(2*np.random.uniform()-1)
                    theta_z = 2*np.pi*np.random.uniform()
                elif self.rotation_type == 'mixedCircleSphereRotation':
                    if idx_rot < self.mixedCircleSphereRotations['nCircleRotations']:                                    
                        rotation = get_circle_rotation(
                            self.mixedCircleSphereRotations['nCircleRotations'], idx_rot)
                    else:
                        theta_x = np.arccos(2*np.random.uniform()-1)
                        theta_z = 2*np.pi*np.random.uniform()            
            # Get dataset from mapping.
            # Get index idx in mapping['indexes'].
            idx_in_mapping = np.where(self.mapping['indexes'] == ID)[0][0]
            # Get dataset from mapping.
            c_dataset = self.mapping['datasets'][idx_in_mapping]            
            # Path dataset.
            suffix_sf = getResampleName(c_dataset, self.resample, self.num_frames)
            pathDataset = os.path.join(self.pathData, '{}dataset{}_{}_{}{}{}{}'.format(self.prefixH5, 
                c_dataset, self.num_frames, self.pose_detector, self.prefix_old_data, self.sensitivity_model, suffix_sf))
            # Adjust idx to match the index in the dataset folder. In that 
            # folder, the first time sequence is indexed 0, whereas
            # mapping['indexes'] is continuous across datasets.
            # Find first index in mapping['indexes'] that is equal to c_dataset.
            idx_in_mapping = np.where(self.mapping['datasets'] == c_dataset)[0][0]
            # Adjust idx.
            ID_adj = ID - idx_in_mapping
                        
            # Load time sequence
            if self.h5:
                with h5py.File(os.path.join(pathDataset, 
                                            'time_sequences.h5'), 'r') as f:
                    grp = f['data']
                    XY_all = {}                          
                    XY_all['features'] = grp['features'][ID_adj, :]
                    XY_all['labels'] = grp['labels'][ID_adj, :] 
            else:
                XY_all = np.load(
                    os.path.join(pathDataset, 
                                 'time_sequence_{}.npy'.format(ID_adj)), 
                    allow_pickle=True).item()
            
            # Process features.
            if self.old_data:
                # 1) Extract indices features.
                idx_features = get_idx_in_all_features_oldData()[0]
                X_temp = XY_all['features'][:,idx_features]                
            else:
                # Does that dataset have arms?
                idx_in_mapping_arms = np.where(
                    np.array(self.mapping['datasets_arms_idx']) == c_dataset)[0][0]
                withArms = self.mapping['datasets_arms_bool'][idx_in_mapping_arms]
                # 1) Extract indices features.
                idx_features, nFeatureMarkers = get_idx_in_all_features(
                    self.augmenter_type, self.pose_detector, 
                    XY_all['features'].shape[1], nDim=3, withArms=withArms, 
                    featureHeight=self.feature_height,
                    featureWeight=self.feature_weight)
                c_features = XY_all['features'][:,idx_features]
                # 2) Express with respect to reference marker.            
                ref_marker_value = get_reference_marker_value(
                    XY_all['features'], self.reference_marker, self.pose_detector, 
                    nDim=3, withArms=withArms)                      
                c_features_wrt_ref = subtract_reference_marker_value(
                    c_features, nFeatureMarkers, ref_marker_value, 
                    featureHeight=self.feature_height, 
                    featureWeight=self.feature_weight)
                # 3) Normalize by subject height.
                height = get_height(XY_all['features'])
                X_temp = normalize_height(
                    c_features_wrt_ref, height, nFeatureMarkers, nDim=3, 
                    featureHeight=self.feature_height,
                    featureWeight=self.feature_weight)            
            # 4) Apply rotation.
            if self.withRotation:
                if self.rotation_type == 'circleRotation':
                    X_temp_xyz_rot = rotateArray(X_temp[:,:-2], 'y', rotation)
                elif self.rotation_type == 'sphereRotation':
                    X_temp_xyz_rot, unit_vec = rotateArraySphere4(
                        X_temp[:,:-2], self.ref_vec, theta_x, theta_z)
                elif self.rotation_type == 'mixedCircleSphereRotation':
                    if idx_rot < self.mixedCircleSphereRotations['nCircleRotations']:
                        X_temp_xyz_rot = rotateArray(X_temp[:,:-2], 'y', rotation)
                    else:
                        X_temp_xyz_rot, unit_vec = rotateArraySphere4(
                            X_temp[:,:-2], self.ref_vec, theta_x, theta_z)
                X_temp_rot = np.concatenate((X_temp_xyz_rot, X_temp[:,-2:]), axis=1)
                X[i,] = X_temp_rot
            else:
                X[i,] = X_temp
            # 5) Add noise.
            if self.noise_bool:
                # Normalize noise magnitude by subject height if not old data.
                if self.old_data:
                    noise_magnitude = self.noise_magnitude
                else:
                    noise_magnitude = self.noise_magnitude/height[0][0]
                noise = np.zeros((self.dim_f[0], self.dim_f[1]))
                noise[:,:self.dim_f[1]-2] = np.random.normal(
                    0, noise_magnitude, (self.dim_f[0], self.dim_f[1]-2))           
                X[i,] += noise
            # 6) Mean subtraction.
            if self.mean_subtraction:                
                X[i,] -= self.features_mean
            # 7) Std division.
            if self.std_normalization:
                X[i,] /= self.features_std
            
            # Process labels.
            if self.old_data:
                # 1) Extract indices labels.       
                idx_labels = get_idx_in_all_labels_oldData(nDim=3)[0]
                y_temp = XY_all['labels'][:,idx_labels]            
            else:
                # 1) Extract indices labels.       
                idx_labels, nResponseMarkers = get_idx_in_all_labels(
                    self.augmenter_type, nDim=3, withArms=withArms)
                c_labels = XY_all['labels'][:,idx_labels]    
                # 2) Express with respect to reference marker.                 
                c_labels_wrt_ref = subtract_reference_marker_value(
                    c_labels, nResponseMarkers, ref_marker_value)
                # 3) Normalize by subject height.
                y_temp = normalize_height(
                    c_labels_wrt_ref, height, nResponseMarkers, nDim=3)            
            # 4) Apply rotation
            if self.withRotation:
                if self.rotation_type == 'circleRotation':
                    y_temp_xyz_rot = rotateArray(y_temp, 'y', rotation)
                elif self.rotation_type == 'sphereRotation':
                    # Use alignment used for features to make sure the same
                    # rotation is applied.
                    y_temp_xyz_rot, _ = rotateArraySphere4(
                        y_temp, self.ref_vec, theta_x, theta_z, unit_vec=unit_vec)
                elif self.rotation_type == 'mixedCircleSphereRotation':
                    if idx_rot < self.mixedCircleSphereRotations['nCircleRotations']:
                        y_temp_xyz_rot = rotateArray(y_temp, 'y', rotation)
                    else:
                        # Use alignment used for features to make sure the same
                        # rotation is applied.
                        y_temp_xyz_rot, _ = rotateArraySphere4(
                            y_temp, self.ref_vec, theta_x, theta_z, unit_vec=unit_vec)
                y[i,] = y_temp_xyz_rot
            else:
                y[i,] = y_temp

        return X, y
    
class dataGeneratorTransformer(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, pathData, batch_size=64, dim_f=(30,59), 
                 dim_r=(30,87), shuffle=True, noise_bool=False, noise_type='',
                 noise_magnitude=0, mean_subtraction=False, 
                 std_normalization=False, features_mean=0, features_std=0,
                 rotation_type='circleRotation', nRotations=0, 
                 ref_vec=np.array([0,0,1]), augmenter_type='lowerExtremity',
                 pose_detector='openpose', feature_height=True, 
                 feature_weight=True, reference_marker='midHip',
                 normalize_data_height=True, mapping={}, num_frames=60,
                 withRotation=False, sensitivity_model='', h5=False,
                 prefixH5='', old_data=False, prefix_old_data='',
                 mixedCircleSphereRotations={}, resample={}):
        'Initialization'
        self.dim_f = dim_f
        self.dim_r = dim_r
        
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        
        self.shuffle = shuffle
        self.on_epoch_end()
        
        self.pathData = pathData
        
        self.noise_bool = noise_bool
        self.noise_magnitude = noise_magnitude 
        
        self.mean_subtraction = mean_subtraction
        self.std_normalization = std_normalization
        self.features_mean = features_mean
        self.features_std = features_std
        
        self.augmenter_type = augmenter_type
        self.pose_detector = pose_detector
        self.sensitivity_model = sensitivity_model
        
        self.h5 = h5
        self.prefixH5 = prefixH5
        
        self.old_data = old_data
        self.prefix_old_data = prefix_old_data
        
        self.feature_height = feature_height
        self.feature_weight = feature_weight
        self.reference_marker = reference_marker
        self.normalize_data_height = normalize_data_height
        
        self.mapping = mapping     
        self.num_frames = num_frames
        
        self.withRotation = withRotation
        self.rotation_type = rotation_type
        self.nRotations = nRotations
        self.ref_vec = ref_vec        
        if not self.withRotation:            
            self.nRotations = 1 # Set to 1 to get proper indexes.
        self.mixedCircleSphereRotations = mixedCircleSphereRotations
        
        self.resample = resample

    def __len__(self):
        'Denotes the number of batches per epoch'        
        # We multiply the number of samples by the number of rotations we want
        # to apply to each sample, and then divide by the batch size.
        return int(np.floor(
            (len(self.list_IDs)*(self.nRotations)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch        
        # Take into account the rotation: we have nRotations x the same sample.
        # Eg,   list_IDs = [0,1,2,3,4,5,6,7,8,9,10,11]
        #       batch_size = 3, n_rot = 4, n_batch = (12*4)/3=16
        #       1st 4 indices correspond to list_IDs0, rot0-4
        #       next 4 indices correspond to list_IDs1, rot0-4
        #       ...
        #       index => floor(index/n_rot)
        #           idx0=0, idx1=0, idx2=0, idx3=0, idx4=1, idx5=1, idx6=1, ...
        index_rot = int(np.floor(index/(self.nRotations)))
        indexes = self.indexes[index_rot*self.batch_size:(index_rot+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]        

        #       ...
        #       index => mod(index, n_rot)
        #           idx0=0, idx1=1, idx2=2, idx3=3, idx4=0, idx5=1, idx6=2, ...
        idx_rot = np.mod(index, (self.nRotations))

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, idx_rot)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, idx_rot):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim_f))
        y = np.empty((self.batch_size, *self.dim_r))
                
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Pick a rotation.
            if self.withRotation:
                if self.rotation_type == 'circleRotation':
                    rotation = get_circle_rotation(self.nRotations, idx_rot)
                elif self.rotation_type == 'sphereRotation':        
                    theta_x = np.arccos(2*np.random.uniform()-1)
                    theta_z = 2*np.pi*np.random.uniform()
                elif self.rotation_type == 'mixedCircleSphereRotation':
                    if idx_rot < self.mixedCircleSphereRotations['nCircleRotations']:                                    
                        rotation = get_circle_rotation(
                            self.mixedCircleSphereRotations['nCircleRotations'], idx_rot)
                    else:
                        theta_x = np.arccos(2*np.random.uniform()-1)
                        theta_z = 2*np.pi*np.random.uniform()            
            # Get dataset from mapping.
            # Get index idx in mapping['indexes'].
            idx_in_mapping = np.where(self.mapping['indexes'] == ID)[0][0]
            # Get dataset from mapping.
            c_dataset = self.mapping['datasets'][idx_in_mapping]            
            # Path dataset.
            suffix_sf = getResampleName(c_dataset, self.resample, self.num_frames)
            pathDataset = os.path.join(self.pathData, '{}dataset{}_{}_{}{}{}{}'.format(self.prefixH5, 
                c_dataset, self.num_frames, self.pose_detector, self.prefix_old_data, self.sensitivity_model, suffix_sf))
            # Adjust idx to match the index in the dataset folder. In that 
            # folder, the first time sequence is indexed 0, whereas
            # mapping['indexes'] is continuous across datasets.
            # Find first index in mapping['indexes'] that is equal to c_dataset.
            idx_in_mapping = np.where(self.mapping['datasets'] == c_dataset)[0][0]
            # Adjust idx.
            ID_adj = ID - idx_in_mapping
                        
            # Load time sequence
            if self.h5:
                with h5py.File(os.path.join(pathDataset, 
                                            'time_sequences.h5'), 'r') as f:
                    grp = f['data']
                    XY_all = {}                          
                    XY_all['features'] = grp['features'][ID_adj, :]
                    XY_all['labels'] = grp['labels'][ID_adj, :] 
            else:
                XY_all = np.load(
                    os.path.join(pathDataset, 
                                 'time_sequence_{}.npy'.format(ID_adj)), 
                    allow_pickle=True).item()
            
            # Process features.
            if self.old_data:
                # 1) Extract indices features.
                idx_features = get_idx_in_all_features_oldData()[0]
                X_temp = XY_all['features'][:,idx_features]                
            else:
                # Does that dataset have arms?
                idx_in_mapping_arms = np.where(
                    np.array(self.mapping['datasets_arms_idx']) == c_dataset)[0][0]
                withArms = self.mapping['datasets_arms_bool'][idx_in_mapping_arms]
                # 1) Extract indices features.
                idx_features, nFeatureMarkers = get_idx_in_all_features(
                    self.augmenter_type, self.pose_detector, 
                    XY_all['features'].shape[1], nDim=3, withArms=withArms, 
                    featureHeight=self.feature_height,
                    featureWeight=self.feature_weight)
                c_features = XY_all['features'][:,idx_features]
                # 2) Express with respect to reference marker.            
                ref_marker_value = get_reference_marker_value(
                    XY_all['features'], self.reference_marker, self.pose_detector, 
                    nDim=3, withArms=withArms)                      
                c_features_wrt_ref = subtract_reference_marker_value(
                    c_features, nFeatureMarkers, ref_marker_value, 
                    featureHeight=self.feature_height, 
                    featureWeight=self.feature_weight)
                # 3) Normalize by subject height.
                height = get_height(XY_all['features'])
                X_temp = normalize_height(
                    c_features_wrt_ref, height, nFeatureMarkers, nDim=3, 
                    featureHeight=self.feature_height,
                    featureWeight=self.feature_weight)            
            # 4) Apply rotation.
            if self.withRotation:
                if self.rotation_type == 'circleRotation':
                    X_temp_xyz_rot = rotateArray(X_temp[:,:-2], 'y', rotation)
                elif self.rotation_type == 'sphereRotation':
                    X_temp_xyz_rot, unit_vec = rotateArraySphere4(
                        X_temp[:,:-2], self.ref_vec, theta_x, theta_z)
                elif self.rotation_type == 'mixedCircleSphereRotation':
                    if idx_rot < self.mixedCircleSphereRotations['nCircleRotations']:
                        X_temp_xyz_rot = rotateArray(X_temp[:,:-2], 'y', rotation)
                    else:
                        X_temp_xyz_rot, unit_vec = rotateArraySphere4(
                            X_temp[:,:-2], self.ref_vec, theta_x, theta_z)
                X_temp_rot = np.concatenate((X_temp_xyz_rot, X_temp[:,-2:]), axis=1)
                X[i,] = X_temp_rot
            else:
                X[i,] = X_temp
            # 5) Add noise.
            if self.noise_bool:
                # Normalize noise magnitude by subject height if not old data.
                if self.old_data:
                    noise_magnitude = self.noise_magnitude
                else:
                    noise_magnitude = self.noise_magnitude/height[0][0]
                noise = np.zeros((self.dim_f[0], self.dim_f[1]))
                noise[:,:self.dim_f[1]-2] = np.random.normal(
                    0, noise_magnitude, (self.dim_f[0], self.dim_f[1]-2))           
                X[i,] += noise
            # 6) Mean subtraction.
            if self.mean_subtraction:                
                X[i,] -= self.features_mean
            # 7) Std division.
            if self.std_normalization:
                X[i,] /= self.features_std
            
            # Process labels.
            if self.old_data:
                # 1) Extract indices labels.       
                idx_labels = get_idx_in_all_labels_oldData(nDim=3)[0]
                y_temp = XY_all['labels'][:,idx_labels]            
            else:
                # 1) Extract indices labels.       
                idx_labels, nResponseMarkers = get_idx_in_all_labels(
                    self.augmenter_type, nDim=3, withArms=withArms)
                c_labels = XY_all['labels'][:,idx_labels]    
                # 2) Express with respect to reference marker.                 
                c_labels_wrt_ref = subtract_reference_marker_value(
                    c_labels, nResponseMarkers, ref_marker_value)
                # 3) Normalize by subject height.
                y_temp = normalize_height(
                    c_labels_wrt_ref, height, nResponseMarkers, nDim=3)            
            # 4) Apply rotation
            if self.withRotation:
                if self.rotation_type == 'circleRotation':
                    y_temp_xyz_rot = rotateArray(y_temp, 'y', rotation)
                elif self.rotation_type == 'sphereRotation':
                    # Use alignment used for features to make sure the same
                    # rotation is applied.
                    y_temp_xyz_rot, _ = rotateArraySphere4(
                        y_temp, self.ref_vec, theta_x, theta_z, unit_vec=unit_vec)
                elif self.rotation_type == 'mixedCircleSphereRotation':
                    if idx_rot < self.mixedCircleSphereRotations['nCircleRotations']:
                        y_temp_xyz_rot = rotateArray(y_temp, 'y', rotation)
                    else:
                        # Use alignment used for features to make sure the same
                        # rotation is applied.
                        y_temp_xyz_rot, _ = rotateArraySphere4(
                            y_temp, self.ref_vec, theta_x, theta_z, unit_vec=unit_vec)
                y[i,] = y_temp_xyz_rot
            else:
                y[i,] = y_temp
                
        return (X[:,:-1,:], y[:,:-1,:]), y[:,1:,:]