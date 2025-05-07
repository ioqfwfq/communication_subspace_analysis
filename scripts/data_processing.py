import numpy as np
from pynwb import NWBHDF5IO

def extract_spikes_from_nwb(file_path, region1='IT', region2='HC'):
    """
    Extract spike times and trial information from NWB file.
    
    Parameters:
    -----------
    file_path : str
        Path to the NWB file
    region1 : str
        Name of the first brain region
    region2 : str
        Name of the second brain region
        
    Returns:
    --------
    spike_times : dict
        Dictionary containing spike times for each neuron
    trial_starts : numpy.ndarray
        Start times for each trial
    trial_ends : numpy.ndarray
        End times for each trial
    trial_types : numpy.ndarray
        Trial type labels
    region1_indices : list
        Indices of neurons in region1
    region2_indices : list
        Indices of neurons in region2
    """
    print(f"Extracting data from {file_path}")
    
    try:
        with NWBHDF5IO(file_path, 'r') as io:
            nwbfile = io.read()
            
            # Extract units (neurons)
            units = nwbfile.units
            
            # Get region labels for each unit
            # NOTE: Adjust this based on your NWB file structure
            # You might need to use 'brain_area', 'region', etc.
            region_labels = units['location'][:]
            
            # Find indices of neurons in each region
            region1_indices = [i for i, region in enumerate(region_labels) if region == region1]
            region2_indices = [i for i, region in enumerate(region_labels) if region == region2]
            
            print(f"Found {len(region1_indices)} neurons in {region1}")
            print(f"Found {len(region2_indices)} neurons in {region2}")
            
            # Extract spike times for each neuron
            spike_times = {}
            for i in region1_indices:
                spike_times[f"{region1}_{i}"] = units['spike_times'][i]
            for i in region2_indices:
                spike_times[f"{region2}_{i}"] = units['spike_times'][i]
            
            # Extract trial information
            trials = nwbfile.trials
            trial_starts = trials['start_time'][:]
            trial_ends = trials['stop_time'][:]
            
            # Trial types might be 'condition', 'trial_type', etc.
            # Adjust based on your NWB structure
            if 'trial_type' in trials.colnames:
                trial_types = trials['trial_type'][:]
            elif 'condition' in trials.colnames:
                trial_types = trials['condition'][:]
            else:
                print("Warning: Could not find trial type information")
                trial_types = np.zeros(len(trial_starts))
                
    except Exception as e:
        print(f"Error reading NWB file: {e}")
        raise
        
    return spike_times, trial_starts, trial_ends, trial_types, region1_indices, region2_indices

def create_spike_count_matrices(spike_times, trial_starts, trial_ends, region1_indices, region2_indices, 
                              region1='IT', region2='HC', bin_size=0.1):
    """
    Create spike count matrices for each region.
    
    Parameters:
    -----------
    spike_times : dict
        Dictionary containing spike times for each neuron
    trial_starts : numpy.ndarray
        Start times for each trial
    trial_ends : numpy.ndarray
        End times for each trial
    region1_indices : list
        Indices of neurons in region1
    region2_indices : list
        Indices of neurons in region2
    region1 : str
        Name of the first brain region
    region2 : str
        Name of the second brain region
    bin_size : float
        Size of time bins in seconds
        
    Returns:
    --------
    X : numpy.ndarray
        Spike counts for region1 (trials × neurons)
    Y : numpy.ndarray
        Spike counts for region2 (trials × neurons)
    """
    n_trials = len(trial_starts)
    n_neurons_region1 = len(region1_indices)
    n_neurons_region2 = len(region2_indices)
    
    # Initialize matrices
    X = np.zeros((n_trials, n_neurons_region1))
    Y = np.zeros((n_trials, n_neurons_region2))
    
    # Fill matrices with spike counts
    for t in range(n_trials):
        start_time = trial_starts[t]
        end_time = trial_ends[t]
        
        # Count spikes for each region1 neuron
        for i, idx in enumerate(region1_indices):
            neuron_key = f"{region1}_{idx}"
            neuron_spikes = spike_times[neuron_key]
            spike_count = np.sum((neuron_spikes >= start_time) & (neuron_spikes <= end_time))
            X[t, i] = spike_count
        
        # Count spikes for each region2 neuron
        for i, idx in enumerate(region2_indices):
            neuron_key = f"{region2}_{idx}"
            neuron_spikes = spike_times[neuron_key]
            spike_count = np.sum((neuron_spikes >= start_time) & (neuron_spikes <= end_time))
            Y[t, i] = spike_count
    
    return X, Y

def preprocess_subject_data(X, Y, trial_types):
    """
    Preprocess data for one subject:
    - Z-score each neuron's activity
    - Compute PSTHs for each trial type
    - Compute residuals by subtracting PSTH
    
    Parameters:
    -----------
    X : numpy.ndarray
        Spike counts for region1 (trials × neurons)
    Y : numpy.ndarray
        Spike counts for region2 (trials × neurons)
    trial_types : numpy.ndarray
        Trial type labels
        
    Returns:
    --------
    X_zscore : numpy.ndarray
        Z-scored activity for region1
    Y_zscore : numpy.ndarray
        Z-scored activity for region2
    X_residual : numpy.ndarray
        Residual activity for region1 after subtracting PSTH
    Y_residual : numpy.ndarray
        Residual activity for region2 after subtracting PSTH
    """
    # Z-score each neuron's activity
    X_zscore = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)  # Adding small constant to avoid division by zero
    Y_zscore = (Y - np.mean(Y, axis=0)) / (np.std(Y, axis=0) + 1e-10)
    
    # Compute PSTHs for each trial type
    unique_trial_types = np.unique(trial_types)
    X_residual = np.zeros_like(X_zscore)
    Y_residual = np.zeros_like(Y_zscore)
    
    for trial_type in unique_trial_types:
        trial_mask = trial_types == trial_type
        
        # Compute mean activity for this trial type (PSTH)
        X_psth = np.mean(X_zscore[trial_mask], axis=0)
        Y_psth = np.mean(Y_zscore[trial_mask], axis=0)
        
        # Subtract PSTH to get residuals
        X_residual[trial_mask] = X_zscore[trial_mask] - X_psth
        Y_residual[trial_mask] = Y_zscore[trial_mask] - Y_psth
    
    return X_zscore, Y_zscore, X_residual, Y_residual