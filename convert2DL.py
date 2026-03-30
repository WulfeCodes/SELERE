import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import torch
import random
import gc
import hashlib

import os
import traceback
import torch.nn.functional as F


#each stride has a kinematic.shape[-1] of trainable windows
class SplitDataset:
    def __init__(self, split,use_noise):
        self.split = split
        self.use_noise = use_noise
        self.data = {
            split: {
                'emg': [],
                'input_kin_state': [],
                'input_gait_pct': [],
                'target_kin_state': [],
                'target_gait_pct': [],
                'target_torque': [],
                'metadata': []
            }
        }

    def __len__(self):
        return len(self.data[self.split]['emg'])

    def verify_lengths(self):
            """Call this after loading all data"""
            lengths = {}
            for key in self.data[self.split].keys():
                lengths[key] = len(self.data[self.split][key])
            
            print(f"\n{self.split.upper()} Dataset Lengths:")
            for key, length in lengths.items():
                print(f"  {key}: {length}")
            
            unique_lengths = set(lengths.values())
            if len(unique_lengths) > 1:
                print(f"  MISMATCH DETECTED! Unique lengths: {unique_lengths}")
                return False
            else:
                print(f"  ✓ All arrays match: {list(unique_lengths)[0]} samples")
                return True

    def __getitem__(self, idx):
        meta = self.data[self.split]['metadata'][idx]
        current_window_idx = meta['window_idx']

        emg_idx = idx
        kin_idx  = idx

        if getattr(self, 'use_noise', False):
            emg_jitter_max = getattr(self, 'emg_jitter_max', 0)
            kin_jitter_max  = getattr(self, 'kin_jitter_max',  0)
            jitter_retries  = getattr(self, 'jitter_retries',  5)

            def sample_valid_idx(base_idx, jitter_max):
                if jitter_max <= 0:
                    return base_idx
                for _ in range(jitter_retries):
                    delta     = random.randint(0, jitter_max)
                    candidate = base_idx - delta
                    if candidate < 0:
                        continue
                    candidate_meta = self.data[self.split]['metadata'][candidate]
                    if (candidate_meta['patient_id'] == meta['patient_id']
                            and candidate_meta['activity']  == meta['activity']
                            and candidate_meta['direction'] == meta['direction']
                            and candidate_meta['window_idx'] == current_window_idx - delta):
                        return candidate
                return base_idx

            emg_idx = sample_valid_idx(idx, emg_jitter_max)
            kin_idx  = sample_valid_idx(idx, kin_jitter_max)

        return {
            'emg':              self.data[self.split]['emg'][emg_idx],
            'input_kin_state':  self.data[self.split]['input_kin_state'][kin_idx],
            'input_gait_pct':   self.data[self.split]['input_gait_pct'][kin_idx],
            'target_kin_state': self.data[self.split]['target_kin_state'][idx],
            'target_gait_pct':  self.data[self.split]['target_gait_pct'][idx],
            'target_torque':    self.data[self.split]['target_torque'][idx],
            'has_torque':       self.data[self.split]['metadata'][idx]['has_torque'],
            'metadata':         self.data[self.split]['metadata'][idx]
        }
    
class WindowedGaitDataParser:
    def __init__(self, window_size=200, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,desired_dataset_size=1000000,
                 input_dir='D:/EMG/postprocessed_datasets',output_dir='D:/EMG/ML_datasets',device='cuda'):
        """
        Gait data parser for impedance-based control system.
        
        Args:
            window_size: Number of EMG samples per window (default 200 at 1000Hz = 0.2s)
            train_ratio, val_ratio, test_ratio: Split ratios (should sum to 1.0)
        """
        self.input_dir = input_dir
        self.output_dir  = output_dir
        self.window_size = window_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.desired_dataset_size = desired_dataset_size
        
        # Main data structure - separated inputs and targets for impedance control

        # Then instantiate:
        self.dataset = {}

        self.parsers = {
            #'lencioni': self.parse_lencioni, #TODO null for step down??
            #'siat': self.parse_siat,
            # 'embry': self.parse_embry,
            #'moreira': self.parse_moreira,
            #'k2muse': self.parse_k2muse,
            # 'angelidou': self.parse_angelidou,

            'bacek' :self.parse_bacek, 
            #'hu': self.parse_hu,

            #'criekinge': self.parse_criekinge,#TODO NaN errors 
           # 'camargo': self.parse_camargo, #TODO NaN errors 

            
            #'moghadam':self.parse_moghadam,
            #'grimmer': self.parse_grimmer, #TODO stride size
            # 'macaluso': self.parse_macaluso,

            #'gait120': self.parse_gait120
        }
        
        # Track patient IDs per dataset for splitting
        self.patient_splits = {}
        self.dataset_patient_counters = defaultdict(int)
        self.dataset_masks = {}

    def save_data(self,dataset_name,activity):

        train_len = len(self.dataset[activity]['train'].data['train']['emg'])
        val_len = len(self.dataset[activity]['val'].data['val']['emg'])
        test_len = len(self.dataset[activity]['test'].data['test']['emg'])
        cumSum = train_len + val_len + test_len
        
        print(f'Finished {dataset_name},{activity}:')
        print(f'  Train ratio: {train_len/cumSum:.3f} ({train_len} samples)')
        print(f'  Val ratio: {val_len/cumSum:.3f} ({val_len} samples)')
        print(f'  Test ratio: {test_len/cumSum:.3f} ({test_len} samples)')
        
        # Add dataset masks (if they exist)
        self.dataset[activity]['train'].data['train']['masks'] = self.dataset_masks[dataset_name]
        self.dataset[activity]['val'].data['val']['masks'] = self.dataset_masks[dataset_name]
        self.dataset[activity]['test'].data['test']['masks'] = self.dataset_masks[dataset_name]
        
        if 'masks' in list(self.dataset[activity]['test'].data['test'].keys()) and 'masks' in list(self.dataset[activity]['val'].data['val'].keys()) and 'masks' in list(self.dataset[activity]['train'].data['train'].keys()):
            print('added masks')
        else: print('error on masks additions') 

        # Stack tensors for each split
        for split_name, dataset in [('train', self.dataset[activity]['train']),
                                    ('val', self.dataset[activity]['val']),
                                    ('test', self.dataset[activity]['test'])]:
            
            split_len = len(dataset.data[split_name]['emg'])
            if split_len == 0:
                print(f'  Skipping {split_name} (empty)')
                continue
            
            print(f'  Stacking {split_name} data...')
            
            for data_type in dataset.data[split_name].keys():
                if data_type not in ['metadata', 'masks','chunk_num'] and isinstance(dataset.data[split_name][data_type], list):
                    if len(dataset.data[split_name][data_type]) > 0:
                        dataset.data[split_name][data_type] = torch.stack(dataset.data[split_name][data_type]).share_memory_()
                        # Free up memory from the list
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Create output directory
        dataset_output_dir = Path(self.output_dir) / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        save_path=Path(dataset_output_dir/activity / str(self.dataset[activity]['chunk_num']))
        save_path.mkdir(parents=True, exist_ok=True)

        
        # Save datasets
        torch.save(self.dataset[activity]['train'].data['train'], dataset_output_dir / activity / str(self.dataset[activity]['chunk_num']) /'train.pt')
        torch.save(self.dataset[activity]['val'].data['val'], dataset_output_dir / activity / str(self.dataset[activity]['chunk_num']) /'val.pt')
        torch.save(self.dataset[activity]['test'].data['test'], dataset_output_dir / activity / str(self.dataset[activity]['chunk_num']) /'test.pt')
        
        self.dataset[activity]['train'] = SplitDataset('train')
        self.dataset[activity]['val'] = SplitDataset('val')
        self.dataset[activity]['test'] = SplitDataset('test')
        self.dataset[activity]['chunk_num']+=1
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print('saved chunk',self.dataset[activity]['chunk_num']-1, 'for dataset: ',dataset_name, 'of activity: ',activity)

    def should_flush(self, activity, dataset_name):

        train_len = len(self.dataset[activity]['train'].data['train']['emg'])
        val_len = len(self.dataset[activity]['val'].data['val']['emg'])
        test_len = len(self.dataset[activity]['test'].data['test']['emg'])
        total = train_len + val_len + test_len
        
        # Hard cap
        if total >= self.desired_dataset_size:
            train_ratio = train_len / total
            val_ratio = val_len / total
            test_ratio = test_len / total

            print('flushing current chunk for ',dataset_name,activity)
            print('total length of current chunk: ', total)
            print(f'train ratio: {train_ratio}, val ratio:: {val_ratio}, test_ratio: {test_ratio}')
            print(f'set train ratio: {self.train_ratio}, set val ratio:: {self.val_ratio}, set test_ratio: {self.test_ratio}')

            return True
        
        # Minimum size

        # Calculate actual ratios


    def get_next_patient_id(self, dataset_name):
        """Get next patient ID for a dataset"""
        patient_id = self.dataset_patient_counters[dataset_name]
        self.dataset_patient_counters[dataset_name] += 1
        return patient_id
    

    def assign_patient_to_split(self, dataset_name, patient_id):
        """Assign a patient to train/val/test split using deterministic hashing"""
        key = f"{dataset_name}_{patient_id}"
        
        if key not in self.patient_splits:
            # Use SHA256 for uniform distribution
            hash_bytes = hashlib.sha256(key.encode()).digest()
            # Convert first 4 bytes to int (0 to 2^32-1)
            hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
            # Normalize to [0, 1) with high precision
            rand_val = hash_int / (2**32)
            
            if rand_val < self.train_ratio:
                split = 'train'
            elif rand_val < self.train_ratio + self.val_ratio:
                split = 'val'
            else:
                split = 'test'
            
            self.patient_splits[key] = split
        
        return self.patient_splits[key]
    
    def compute_omega(self, angles, idx, dt=0.005):
        """
        Compute angular velocity via finite difference.
        
        Args:
            angles: (n_samples, n_joints) array of joint angles
            idx: current index
            dt: time step (1/200 Hz = 0.005s for gait-normalized data)
        
        Returns:
            omega: (n_joints,) angular velocities
        """
        if idx == 0:
            # Forward difference for first point
            omega = (angles[:,:,idx + 1] - angles[:,:,idx]) / dt
        else:
            # Backward difference (causal, uses only past data)
            omega = (angles[:,:,idx] - angles[:,:,idx - 1]) / dt
        return omega
    
    def compute_alpha(self, angles, idx, dt=0.005):
        """
        Compute angular acceleration via second-order finite difference.
        
        Args:
            angles: (n_joints,n_axis,n_samples) array of joint angles
            idx: current index
            dt: time step (1/200 Hz = 0.005s)
        
        Returns:
            alpha: (n_joints,) angular accelerations
        """
        if idx == 0:
            # Use first two points for initial acceleration
            omega_curr = (angles[:,:,1] - angles[:,:,0]) / dt
            omega_next = (angles[:,:,2] - angles[:,:,1]) / dt
            alpha = (omega_next - omega_curr) / dt
        elif idx == angles.shape[-1] - 1:
            # Use last two points
            omega_prev = (angles[:,:,idx-1] - angles[:,:,idx - 2]) / dt
            omega_curr = (angles[:,:,idx] - angles[:,:,idx-1]) / dt
            alpha = (omega_curr - omega_prev) / dt
        else:
            # Central difference for better accuracy
            omega_prev = (angles[:,:,idx] - angles[:,:,idx - 1]) / dt
            omega_next = (angles[:,:,idx + 1] - angles[:,:,idx]) / dt
            alpha = (omega_next - omega_prev) / dt
        return alpha
    
    def extract_windows_aligned_to_kinematics(self, stride_emg, stride_kin, stride_kinetic, stride_gait_pct):
        """
        Extract windows aligned to kinematic sampling for impedance control.
        Each window predicts the NEXT kinematic state and impedance parameters.
        
        Args:
            stride_emg: (n_emg_samples, n_emg_channels) - e.g., 1000 samples at 1000Hz
            stride_kin: (200, 9) - joint angles at 200 gait-normalized points
            stride_kinetic: (200, 9) - joint torques or None
            stride_gait_pct: (200,) - gait percentages [0.0, 0.005, ..., 1.0]
        
        Returns:
            List of dictionaries containing aligned windows

        """
        if stride_emg is None or stride_kin is None or stride_emg.shape[-1] < self.window_size:
            return []
                        
        n_emg_samples = stride_emg.shape[-1]
        n_kin_samples = stride_kin.shape[-1]
        
        windows = []
        
        # For each kinematic point (except first - no previous state)
        for kin_idx in range(1, n_kin_samples):
            # Map kinematic index to EMG timeline
            emg_idx = int((kin_idx / n_kin_samples) * n_emg_samples)
            
            # Extract EMG window ending at this point
            emg_start = emg_idx - self.window_size
            emg_end = emg_idx
            
            #pad if not enough EMG history
            if emg_start < 0:
                pad_size = -emg_start
                # stride_emg is (channels, time), slice time dimension
                emg_window = F.pad(
                    stride_emg[:, 0:emg_end],
                    (pad_size, 0),  # Pad time dimension (axis=1)
                    mode='replicate'
                )

            else:
                emg_window = stride_emg[:,emg_start:emg_end]
            
            # Previous kinematic state (INPUT - current actual state)
            prev_angles = stride_kin[:,:,kin_idx - 1]  # (9,)
            prev_omega = self.compute_omega(stride_kin, kin_idx - 1,dt=1/stride_kin.shape[-1])  # (9,)
            prev_alpha = self.compute_alpha(stride_kin, kin_idx - 1,dt=1/stride_kin.shape[-1])  # (9,)
            input_kin_state = torch.concatenate([prev_angles.flatten(), prev_omega.flatten(), prev_alpha.flatten()])  # (27,)
            
            # Current kinematic state (TARGET - desired next state)
            curr_angles = stride_kin[:,:,kin_idx]  # (9,)
            curr_omega = self.compute_omega(stride_kin, kin_idx,dt=1/stride_kin.shape[-1])  # (9,)
            curr_alpha = self.compute_alpha(stride_kin, kin_idx,dt=1/stride_kin.shape[-1])  # (9,)
            target_kin_state = torch.cat([curr_angles.flatten(), curr_omega.flatten(), curr_alpha.flatten()])  # (27,)
            
            # Gait percentages
            input_gait_pct = torch.Tensor(stride_gait_pct[kin_idx - 1]).unsqueeze(0)
            target_gait_pct = torch.Tensor(stride_gait_pct[kin_idx])
            
            # Target torque for impedance loss (if available)
            target_torque = stride_kinetic[:,:,kin_idx].flatten() if stride_kinetic is not None and stride_kinetic.any() else None    

            windows.append({
                'emg': emg_window,                      # (100, 13)
                'input_kin_state': input_kin_state,     # (27,) - current state
                'input_gait_pct': input_gait_pct,       # scalar
                'target_kin_state': target_kin_state,   # (27,) - desired state
                'target_gait_pct': target_gait_pct,     # scalar
                'target_torque': target_torque          # (9,) or None
            })
        
        return windows
    
    def add_stride(self, anthro,stride_emg, stride_kin, stride_kinetic, stride_gait_pct,
                   activity, direction, patient_id, dataset_name):
        """
        Process a stride and add windowed samples to appropriate split.
        
        Args:
            stride_emg: EMG data (n_samples, n_emg_channels)
            stride_kin: Kinematic angles (200, 9)
            stride_kinetic: Kinetic torques (200, 9) or None
            stride_gait_pct: Gait percentage array (200,)
            activity: Activity type (e.g., 'walk', 'stair_up')
            direction: Direction ('left', 'right', etc.)
            patient_id: Patient identifier
            dataset_name: Name of the dataset
        """


        if activity not in self.dataset.keys():
            print('entered')
            self.dataset[activity] = {'train':SplitDataset('train'),
                                        'val':SplitDataset('val'),
                                        'test':SplitDataset('test'),
                                        'chunk_num': 0}

        # Determine which split this patient belongs to
        split = self.assign_patient_to_split(dataset_name, patient_id)
        
        # Extract windows aligned to kinematic sampling
        windows = self.extract_windows_aligned_to_kinematics(
            stride_emg, stride_kin, stride_kinetic, stride_gait_pct
        )
        
        if len(windows) == 0:
            print('ERROR W WINDOW EXTRACTION')
            return
        
        # Add each window to the appropriate split
        for window_idx, window in enumerate(windows):
            # Map split to dataset (this is just a reference/pointer, no copying)
            if split == 'train':
                curr_dataset = self.dataset[activity]['train']
            elif split == 'val':
                curr_dataset = self.dataset[activity]['val']
            elif split == 'test':
                curr_dataset = self.dataset[activity]['test']

            # Now append to the current dataset

            curr_dataset.data[split]['emg'].append(window['emg'])
            curr_dataset.data[split]['input_kin_state'].append(window['input_kin_state'])
            curr_dataset.data[split]['input_gait_pct'].append(window['input_gait_pct'])
            curr_dataset.data[split]['target_kin_state'].append(window['target_kin_state'])
            curr_dataset.data[split]['target_gait_pct'].append(window['target_gait_pct'])

            # Add torque or placeholder
            #for Hu, Bacek, and gait120
            if window['target_torque'] is not None and window['target_torque'].any():
                curr_dataset.data[split]['target_torque'].append(window['target_torque'])
            else:
                curr_dataset.data[split]['target_torque'].append(torch.Tensor(np.zeros(9)).share_memory_())

            # Add metadata
            metadata = {
                'activity': activity,
                'direction': direction,
                'patient_id': patient_id,
                'dataset': dataset_name,
                'window_idx': window_idx,
                'has_torque': bool(window['target_torque'] is not None and window['target_torque'].any()),
                'anthro':anthro,
            }
            curr_dataset.data[split]['metadata'].append(metadata)

            if self.should_flush(activity=activity,dataset_name=dataset_name):
                print('choosing to flush')
                self.save_data(dataset_name=dataset_name,activity=activity)

    
    def extract_masks(self, data, dataset_name):
        """Extract masks from pickle file based on dataset-specific structure"""

        masks = {}
        
        # Dataset-specific mask extraction
        if dataset_name == 'criekinge':
                masks['emg'] = torch.Tensor(data['mask']['emg'])
                masks['kinematic'] = torch.Tensor(data['mask']['angle'])
                masks['kinetic'] = torch.Tensor(data['mask']['kinetics']) 
                    
        elif dataset_name == 'moghadam':
            masks['emg'] = torch.Tensor(data['mask']['left']['emg'])
            masks['kinematic'] = torch.Tensor(data['mask']['left']['kinematic'])
            masks['kinetic'] = torch.Tensor(data['mask']['left']['kinetic']) 
            
        elif dataset_name == 'lencioni':
            masks['emg'] = torch.Tensor(data['mask']['emg'])
            masks['kinematic'] = torch.Tensor(data['mask']['angle']) 
            masks['kinetic'] = torch.Tensor(data['mask']['kinetic']) 
            
        elif dataset_name == 'moreira':
            masks['emg'] = torch.Tensor(data['mask']['left']['emg'])
            masks['kinematic'] = torch.Tensor(data['mask']['left']['angle'])
            masks['kinetic'] = torch.Tensor(data['mask']['left']['kinetic'])
            
        elif dataset_name == 'hu':
            masks['emg'] = torch.Tensor(data['masks']['left']['emg'])
            masks['kinematic'] = torch.Tensor(data['masks']['left']['angles'])
            masks['kinetic'] = None  # No kinetic data in hu
            
        elif dataset_name == 'grimmer':
            masks['emg'] = torch.Tensor(data['mask']['left']['emg'])
            masks['kinematic'] = torch.Tensor(data['mask']['left']['angle'])
            masks['kinetic'] = torch.Tensor(data['mask']['left']['kinetic'])
            
        elif dataset_name == 'siat':
            masks['emg'] = torch.Tensor(data['masks']['left']['emg'])
            masks['kinematic'] = torch.Tensor(data['masks']['left']['angle'])
            masks['kinetic'] = torch.Tensor(data['masks']['left']['kinetic']) 
            
        elif dataset_name == 'embry':
            masks['emg'] = torch.Tensor(data['mask']['left']['emg'])
            masks['kinematic'] = torch.Tensor(data['mask']['left']['kinematic'])
            masks['kinetic'] = torch.Tensor(data['mask']['left']['kinetic']) 
            
        elif dataset_name == 'gait120':
            masks['emg'] = torch.Tensor(data['mask']['emg'])
            masks['kinematic'] = torch.Tensor(data['mask']['angle'])
            masks['kinetic'] = None  # No kinetic data in gait120
            
        elif dataset_name == 'camargo':
            masks['emg'] = torch.Tensor(data['mask']['emg'])
            masks['kinematic'] = torch.Tensor(data['mask']['angle'])
            masks['kinetic'] = torch.Tensor(data['mask']['kinetic']) 
            
        elif dataset_name == 'angelidou':
            masks['emg'] = torch.Tensor(data['mask']['left']['emg'])
            masks['kinematic'] = torch.Tensor(data['mask']['left']['angle'])
            masks['kinetic'] = torch.Tensor(data['mask']['left']['kinetic']) 
            
        elif dataset_name == 'bacek':

            # Bacek stores masks directly in the walk data structure
            masks['emg'] = torch.Tensor(data['mask']['right']['emg'])
            masks['kinematic'] = torch.Tensor(data['mask']['right']['angle'])
            masks['kinetic'] = None  # No kinetic data in bacek
            
        elif dataset_name == 'macaluso':
            masks['emg'] = torch.Tensor(data['mask']['left']['emg'])
            masks['kinematic'] = torch.Tensor(data['mask']['left']['kinematic'])
            masks['kinetic'] = torch.Tensor(data['mask']['left']['kinetic']) 
            
        elif dataset_name == 'k2muse':
            # K2muse has different channels for left/right direction
            K2museRightEMGs = ['VLO', 'RF', 'VMO', 'TA', 'BF','SEM', 'MG', 'ML', 'SOL',0,0,0,0]
            K2museLeftEMGs =  [0,'RF', 0,'TA','BF',0,'LG',0,0,0,0,0,0]
            
            masks['emg'] = torch.Tensor([1 if ch != 0 else 0 for ch in K2museRightEMGs])
            
            masks['kinematic'] = torch.Tensor(data['mask']['right']['angle'])
            masks['kinetic'] = torch.Tensor(data['mask']['right']['kinetic']) 
                
        return masks
    
    def get_split_stats(self):
            """Get statistics about the splits"""
            dataset_map = {
                'train': self.trainDataset,
                'val': self.valDataset,
                'test': self.testDataset
            }
            
            stats = {}
            for split in ['train', 'val', 'test']:
                curr_dataset = dataset_map[split]
                n_windows = len(curr_dataset.data[split]['emg'])
                n_with_torque = sum(meta['has_torque'] for meta in curr_dataset.data[split]['metadata'])
                
                stats[split] = {
                    'n_windows': n_windows,
                    'n_with_torque': n_with_torque,
                    'n_without_torque': n_windows - n_with_torque
                }
            
            return stats
    
    def parse_moghadam(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['moghadam'] = self.extract_masks(data, 'moghadam')
        
        directions = ['left', 'right']
        
        for leg in directions:
            # Zip all data and wrap with tqdm
            patients = list(zip(
                data['walk'][leg]['emg'],
                data['walk'][leg]['kinematic'],
                data['walk'][leg]['kinetic'],
                data['walk'][leg]['emg_gait_percentage']
            ))
            
            for pat_emg, pat_kin, pat_kinetic, pat_gait in tqdm(
                patients,
                desc=f"Moghadam {leg}",
                unit="patient"
            ):
                patient_id = self.get_next_patient_id('moghadam')
                
                for trial_emg, trial_kin, trial_kinetic, trial_gait in zip(pat_emg, pat_kin, pat_kinetic, pat_gait):
                    if len(trial_emg) == 0 or len(trial_kin) == 0 or len(trial_kinetic) == 0 or len(trial_gait) == 0:
                        continue
                    for stride_emg, stride_kin, stride_kinetic, stride_gait in zip(trial_emg, trial_kin, trial_kinetic, trial_gait):
                        self.add_stride(data['metadata'][patient_id],stride_emg, stride_kin, stride_kinetic, stride_gait,
                                    'walk', leg, patient_id, 'moghadam')


    def parse_criekinge(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['criekinge'] = self.extract_masks(data, 'criekinge')
        
        directions = ['left', 'right', 'stroke']
        
        for leg in directions:
            patients = list(zip(
                data['walk'][leg]['emg'],
                data['walk'][leg]['angle'],
                data['walk'][leg]['kinetics'],
                data['walk'][leg]['emg_gait_percentage']
            ))
            
            for pat_emg, pat_kin, pat_kinetic, pat_gait_pct in tqdm(
                patients,
                desc=f"Criekinge {leg}",
                unit="patient"
            ):
                patient_id = self.get_next_patient_id('criekinge')
                
                for stride_emg, stride_kin, stride_kinetic, stride_gait_pct in zip(pat_emg, pat_kin, pat_kinetic, pat_gait_pct):
                    self.add_stride(data['metadata'][patient_id],stride_emg, stride_kin, stride_kinetic, stride_gait_pct,
                                'walk', leg, patient_id, 'criekinge')
                    

    def parse_lencioni(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(data.keys())
        
        self.dataset_masks['lencioni'] = self.extract_masks(data, 'lencioni')
        
        activities = ['step up', 'walk'] #TODO ['step up', 'step down ,'walk']
        
        for activity in activities:
            print('lencioni activity',activity,activity not in self.dataset.keys(),activity not in list(self.dataset.keys()),activity=='step down')#TODO
            patients = list(zip( 
                data[activity]['emg'],
                data[activity]['angle'],
                data[activity]['kinetic'],
                data[activity]['emg_gait_percentage']
            ))

            
            for pat_emg, pat_kin, pat_kinetic, pat_gait_pct in tqdm(
                patients,
                desc=f"Lencioni {activity}",
                unit="patient"
            ):

                patient_id = self.get_next_patient_id('lencioni')
                
                for stride_emg, stride_kin, stride_kinetic, stride_gait_pct in zip(pat_emg, pat_kin, pat_kinetic, pat_gait_pct):
                    if stride_gait_pct is None:
                            print(f"Warning: stride_gait_pct is None")
                    if stride_kin is None:
                        print(f"Warning: stride_kin is None")
                    if stride_kinetic is None:
                        print(f"Warning: stride_kinetic is None")
                    self.add_stride(data['metadata'][patient_id],stride_emg, stride_kin, stride_kinetic, stride_gait_pct,
                                activity, 'unknown', patient_id, 'lencioni')
                    

    def parse_moreira(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['moreira'] = self.extract_masks(data, 'moreira')
        
        directions = ['left', 'right']
        
        for direction in directions:
            # Zip all data and wrap with tqdm
            patients = list(zip(
                data['walk'][direction]['emg'],
                data['walk'][direction]['angle'],
                data['walk'][direction]['kinetic'],
                data['walk'][direction]['emg_gait_percentage']
            ))
            
            for pat_emg, pat_kin, pat_kinetic, pat_gait_pct in tqdm(
                patients, 
                desc=f"Moreira {direction} patients", 
                unit="patient"
            ):
                patient_id = self.get_next_patient_id('moreira')
                
                for trial_emg, trial_kin, trial_kinetic, trial_gait_pct in zip(pat_emg, pat_kin, pat_kinetic, pat_gait_pct):
                    for stride_emg, stride_kin, stride_kinetic, stride_gait_pct in zip(trial_emg, trial_kin, trial_kinetic, trial_gait_pct):
                        self.add_stride(data['metadata'][patient_id],stride_emg, stride_kin, stride_kinetic, stride_gait_pct,
                                    'walk', direction, patient_id, 'moreira')
                        
    def parse_hu(self, pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            self.dataset_masks['hu'] = self.extract_masks(data, 'hu')
            
            activities = ['walk', 'ramp_up', 'ramp_down', 'stair_up', 'stair_down']
            directions = ['right','left']
            
            for activity in activities:
                for direction in directions:
                    patients = list(zip(
                        data[activity][direction]['emg'],
                        data[activity][direction]['angle'],
                        data[activity][direction]['emg_gait_percentage']
                    ))
                    
                    for pat_emg, pat_kin, pat_gait_pct in tqdm(
                        patients,
                        desc=f"Hu {activity} {direction}",
                        unit="patient"
                    ):
                        patient_id = self.get_next_patient_id('hu')
                        
                        for stride_emg, stride_kin, stride_gait_pct in zip(pat_emg, pat_kin, pat_gait_pct):
                            self.add_stride(data['metadata'][patient_id],stride_emg, stride_kin, None, stride_gait_pct,
                                        activity, direction, patient_id, 'hu')
        except Exception as e:
            print(data.keys())
                        
    def parse_grimmer(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['grimmer'] = self.extract_masks(data, 'grimmer')
        
        activities = ['stairUp', 'stairDown']
        directions = ['left', 'right']
        
        for activity in activities:
            for direction in directions:
                patients = list(zip(
                    data[activity][direction]['emg'],
                    data[activity][direction]['angle'],
                    data[activity][direction]['kinetic'],
                    data[activity][direction]['emg_gait_percentage']
                ))
                
                for pat_emg, pat_kin, pat_kinetic, pat_gait_pct in tqdm(
                    patients,
                    desc=f"Grimmer {activity} {direction}",
                    unit="patient"
                ):
                    patient_id = self.get_next_patient_id('grimmer')
                    
                    for trial_emg, trial_kin, trial_kinetic, trial_gait_pct in zip(pat_emg, pat_kin, pat_kinetic, pat_gait_pct):
                        for stride_emg, stride_kin, stride_kinetic, stride_gait_pct in zip(trial_emg, trial_kin, trial_kinetic, trial_gait_pct):
                            self.add_stride(data['metadata'][patient_id],stride_emg, stride_kin, stride_kinetic, stride_gait_pct,
                                        activity, direction, patient_id, 'grimmer')
                            
    def parse_siat(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['siat'] = self.extract_masks(data, 'siat')
        
        activities = ['walk', 'stair_up', 'stair_down']
        
        for activity in activities:
            patients = list(zip(
                data[activity]['left']['emg'],
                data[activity]['left']['angle'],
                data[activity]['left']['kinetic'],
                data[activity]['left']['emg_gait_percentage']
            ))
            
            for pat_emg, pat_kin, pat_kinetic, pat_gait_pct in tqdm(
                patients,
                desc=f"Siat {activity}",
                unit="patient"
            ):
                patient_id = self.get_next_patient_id('siat')
                
                for session_emg, session_kin, session_kinetic, session_gait_pct in zip(pat_emg, pat_kin, pat_kinetic, pat_gait_pct):
                    for stride_emg, stride_kin, stride_kinetic, stride_gait_pct in zip(session_emg, session_kin, session_kinetic, session_gait_pct):
                        self.add_stride(data['metadata'][patient_id],stride_emg, stride_kin, stride_kinetic, stride_gait_pct,
                                    activity, 'left', patient_id, 'siat')
                        
    def parse_embry(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['embry'] = self.extract_masks(data, 'embry')
        
        activities = ['rampdown']#['walk', 'rampup', 'rampdown']#TODO verify split stats
        directions = ['left', 'right']
        
        for direction in directions:
            for activity in activities:
                patients = list(zip(
                    data[activity][direction]['emg'],
                    data[activity][direction]['kinematic'],
                    data[activity][direction]['kinetic'],
                    data[activity][direction]['emg_gait_percentage']
                ))
                
                for pat_emg, pat_kin, pat_kinetic, pat_gait_pct in tqdm(
                    patients,
                    desc=f"Embry {direction} {activity}",
                    unit="patient"
                ):
                    patient_id = self.get_next_patient_id('embry')
                    
                    for trial_emg, trial_kin, trial_kinetic, trial_gait_pct in zip(pat_emg, pat_kin, pat_kinetic, pat_gait_pct):
                        for stride_emg, stride_kin, stride_kinetic, stride_gait_pct in zip(trial_emg, trial_kin, trial_kinetic, trial_gait_pct):
                            self.add_stride(data['metadata'][patient_id],stride_emg, stride_kin, stride_kinetic, stride_gait_pct,
                                        activity, direction, patient_id, 'embry')
                            
    def parse_gait120(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['gait120'] = self.extract_masks(data, 'gait120')
        
        activities = ['sitToStand', 'standToSit', 'levelWalking', 'stairAscent', 'stairDescent', 'slopeAscent', 'slopeDescent']
        
        for activity in activities:
            patients = list(zip(
                data['right'][activity]['emg'],
                data['right'][activity]['angle'],
                data['right'][activity]['emg_gait_percentage']
            ))
            
            for pat_emg, pat_kin, pat_gait_pct in tqdm(
                patients,
                desc=f"Gait120 {activity}",
                unit="patient"
            ):
                patient_id = self.get_next_patient_id('gait120')
                
                for stride_emg, stride_kin, stride_gait_pct in zip(pat_emg, pat_kin, pat_gait_pct):
                    self.add_stride(data['metadata'][patient_id],stride_emg, stride_kin, None, stride_gait_pct,
                                activity, 'right', patient_id, 'gait120')

    def parse_camargo(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['camargo'] = self.extract_masks(data, 'camargo')
        
        activities = ['walk', 'stair', 'ramp']
        
        for activity in activities:
            patients = list(zip(
                data['right'][activity]['emg'],
                data['right'][activity]['angle'],
                data['right'][activity]['kinetic'],
                data['right'][activity]['emg_gait_percentage']
            ))
            
            for pat_emg, pat_kin, pat_kinetic, pat_gait_pct in tqdm(
                patients,
                desc=f"Camargo {activity}",
                unit="patient"
            ):
                patient_id = self.get_next_patient_id('camargo')
                
                for trial_emg, trial_kin, trial_kinetic, trial_gait_pct in zip(pat_emg, pat_kin, pat_kinetic, pat_gait_pct):
                    for stride_emg, stride_kin, stride_kinetic, stride_gait_pct in zip(trial_emg, trial_kin, trial_kinetic, trial_gait_pct):
                        self.add_stride(data['metadata'][patient_id],stride_emg, stride_kin, stride_kinetic, stride_gait_pct,
                                    activity, 'right', patient_id, 'camargo')
                        

    def parse_angelidou(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['angelidou'] = self.extract_masks(data, 'angelidou')
        
        directions = ['left', 'right']
        
        for direction in directions:
            patients = list(zip(
                data['walk'][direction]['emg'],
                data['walk'][direction]['angle'],
                data['walk'][direction]['kinetic'],
                data['walk'][direction]['emg_gait_percentage']
            ))
            
            for pat_emg, pat_kin, pat_kinetic, pat_gait_pct in tqdm(
                patients,
                desc=f"Angelidou {direction}",
                unit="patient"
            ):
                patient_id = self.get_next_patient_id('angelidou')
                
                for stride_emg, stride_kin, stride_kinetic, stride_gait_pct in zip(pat_emg, pat_kin, pat_kinetic, pat_gait_pct):
                    self.add_stride(data['metadata'][patient_id],stride_emg, stride_kin, stride_kinetic, stride_gait_pct,
                                'walk', direction, patient_id, 'angelidou')
                    

    def parse_bacek(self, pkl_path):

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        metaPatientInfo=torch.load('C:/EMG/bacekNames')
        
        self.dataset_masks['bacek'] = self.extract_masks(data, 'bacek')
        
        directions = ['left', 'right']
        
        for direction in directions:
            patients = list(zip(
                data['walk'][direction]['emg'],
                data['walk'][direction]['angle'],
                data['walk'][direction]['emg_gait_percentage']
            ))
            
            for pat_emg, pat_kin, pat_gait_pct in tqdm(
                patients,
                desc=f"Bacek {direction}",
                unit="patient"
            ):
                patient_id = self.get_next_patient_id('bacek')
                
                for trial_emg, trial_kin, trial_gait_pct in zip(pat_emg, pat_kin, pat_gait_pct):
                    for stride_emg, stride_kin, stride_gait_pct in zip(trial_emg, trial_kin, trial_gait_pct):
                        self.add_stride(metaPatientInfo[patient_id],stride_emg, stride_kin, None, stride_gait_pct,
                                    'walk', direction, patient_id, 'bacek')
                        
    def parse_macaluso(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['macaluso'] = self.extract_masks(data, 'macaluso')
        
        activities = ['walk', 'rampup', 'rampdown']
        directions = ['left', 'right']
        
        for activity in activities:
            for direction in directions:
                patients = list(zip(
                    data[activity][direction]['emg'],
                    data[activity][direction]['kinematic'],
                    data[activity][direction]['kinetic'],
                    data[activity][direction]['emg_gait_percentage']
                ))
                
                for pat_emg, pat_kin, pat_kinetic, pat_gait_pct in tqdm(
                    patients,
                    desc=f"Macaluso {activity} {direction}",
                    unit="patient"
                ):
                    patient_id = self.get_next_patient_id('macaluso')
                    
                    for trial_emg, trial_kin, trial_kinetic, trial_gait_pct in zip(pat_emg, pat_kin, pat_kinetic, pat_gait_pct):
                        for stride_emg, stride_kin, stride_kinetic, stride_gait_pct in zip(trial_emg, trial_kin, trial_kinetic, trial_gait_pct):
                            self.add_stride(data['metadata'][patient_id],stride_emg, stride_kin, stride_kinetic, stride_gait_pct,
                                        activity, direction, patient_id, 'macaluso')
                        
        
    def parse_k2muse(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['k2muse'] = self.extract_masks(data, 'k2muse')
        
        activities = ['walk', 'up_ramp', 'down_ramp']
        
        for activity in activities:
            patients = list(zip(
                data['right'][activity]['emg'],
                data['right'][activity]['angle'],
                data['right'][activity]['kinetic'],
                data['right'][activity]['emg_gait_percentage']
            ))
            
            for pat_emg, pat_kin, pat_kinetic, pat_gait_pct in tqdm(
                patients,
                desc=f"K2muse {activity}",
                unit="patient"
            ):
                patient_id = self.get_next_patient_id('k2muse')
                
                for trial_emg, trial_kin, trial_kinetic, trial_gait_pct in zip(pat_emg, pat_kin, pat_kinetic, pat_gait_pct):
                    for subtrial_emg, subtrial_kin, subtrial_kinetic, subtrial_gait_pct in zip(trial_emg, trial_kin, trial_kinetic, trial_gait_pct):
                        for stride_emg, stride_kin, stride_kinetic, stride_gait_pct in zip(subtrial_emg, subtrial_kin, subtrial_kinetic, subtrial_gait_pct):
                            self.add_stride(data['metadata'][patient_id],stride_emg, stride_kin, stride_kinetic, stride_gait_pct,
                                        activity, 'right', patient_id, 'k2muse')
                            

    def convert_all(self):
        """Convert all datasets to .tfS"""
        print("Starting conversion of all datasets...")
        
        for dataset_name, parser_func in tqdm(self.parsers.items(), desc="Processing datasets"):
            pkl_path = Path(f"{self.input_dir}/{dataset_name}.pkl")
            if pkl_path.exists():
                print(f"\nProcessing {dataset_name}...")
                try:
                    parser_func(pkl_path)
                except Exception as e:
                    print(f"  Error processing {dataset_name}: {e}")
                    traceback.print_exc()
                cumSum=len(self.data['train']['emg']) + len(self.data['val']['emg']) + len(self.data['test']['emg'])
                #print(f'finished {dataset_name},\ntrain ratio: {len(self.data['train']['emg'])/cumSum},\ntest ratio: {len(self.data['test']['emg'])/cumSum}\n,val ration: {len(self.data['val']['emg'])/cumSum}')
            else:
                print(f"\nWarning: {pkl_path} not found, skipping...")


def export_all(window_size=None, train_ratio=None, val_ratio=None, test_ratio=None, 
               output_path='D:/EMG/ML_datasets'):
    
    sampleParser = WindowedGaitDataParser(
        window_size=window_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        desired_dataset_size=1000000,
        output_dir=output_path,
    )

    for dataset_name, parser_func in tqdm(sampleParser.parsers.items(), desc="Processing datasets"):
        pkl_path = Path(f"{sampleParser.input_dir}/{dataset_name}.pkl")
        
        if not pkl_path.exists():
            print(f"\nWarning: {pkl_path} not found, skipping...")
            continue
                    
        print(f"\nProcessing {dataset_name}...")


        try:
            parser_func(pkl_path)
            
            # Calculate and print ratios
            for activity in list(sampleParser.dataset.keys()):
                sampleParser.save_data(dataset_name=dataset_name,activity=activity)
                
                             
        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
        finally:
            sampleParser.dataset = {}
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print('garbage collected and finished dataset')
                
# Example usage and helper functions:
def main():

    export_all(window_size=100,train_ratio=.70, val_ratio=.20,test_ratio=.10)
    
if __name__ == "__main__":
    main()