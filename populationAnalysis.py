# %% [markdown]
# # Population Analysis of Neural Data
# 
# This notebook performs population-level analyses on neural recordings from the hippocampus and amygdala during memory tasks. The analyses include:
#
# - Computing population activity patterns and trial-averaged responses
# - Identifying communication subspaces between brain regions using reduced rank regression
# - Performing control analyses with shuffled data and within-region comparisons
# - Visualizing results with cross-validated performance metrics
#
# The data is stored in Neurodata Without Borders (NWB) format, which provides standardized organization of neurophysiology data.
#
# * Author: Junda Zhu
# %%
import numpy as np
from pathlib import Path
import pandas as pd
import os
import sys
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
#from nwbwidgets import nwb2widget
import seaborn as sns
import RutishauserLabtoNWB.events.newolddelay.python.analysis.helper as helper
import RutishauserLabtoNWB.events.newolddelay.python.analysis.single_neuron as single_neuron

# %%
# --- all neuron analysis Load all NWB files and merge all neuron objects into all_neurons
nwb_dir = Path('data')
nwb_files = sorted(list(nwb_dir.glob('*.nwb')))
print(f"[END] Found {len(nwb_files)} NWB files:", nwb_files)

all_neurons = []
brainAreas_all = []
neuron_session_idx = []  # Track session index for each neuron
for session_idx, nwb_path in enumerate(nwb_files):
    print(f"[END] Loading {nwb_path}")
    io = NWBHDF5IO(str(nwb_path), mode='r')
    nwb = io.read()
    neurons_this_file = single_neuron.extract_neuron_data_from_nwb(nwb)
    electrode_areas = np.asarray(nwb.electrodes['location'].data)
    try:
        unit_electrodes = np.asarray(nwb.units['electrodes'].target.data)
    except:
        unit_electrodes = np.asarray(nwb.units['electrodes'].data)
    unit_areas = electrode_areas[unit_electrodes]
    brainAreas_all = np.append(brainAreas_all, unit_areas)
    all_neurons.extend(neurons_this_file)
    neuron_session_idx.extend([session_idx] * len(neurons_this_file))
    io.close()

print(f"[END] Total neurons loaded: {len(all_neurons)}")
print(f"[END] Unique brain areas: {set(brainAreas_all)}")

# Use the first NWB file to get trial indices for pooled analysis
io = NWBHDF5IO(str(nwb_files[0]), mode='r')
nwb = io.read()
trial_fields = nwb.trials.colnames
print(trial_fields)
stim_phase = nwb.trials['stim_phase'].data[:]
new_old_labels = nwb.trials['new_old_labels_recog'].data[:]
encoding_trial_indices = [i for i, phase in enumerate(stim_phase) if phase == b'learn']
first_presentation_indices = [i for i in encoding_trial_indices if new_old_labels[i] == b'NA']
trial_starts = nwb.trials['start_time'].data[:]
trial_types = nwb.trials['stimCategory'].data[first_presentation_indices]
io.close()

# Step 1: Identify HC and AMY neuron indices in all_neurons
hc_indices_all = [i for i, area in enumerate(brainAreas_all) if b'Hippocampus' in area]
amy_indices_all = [i for i, area in enumerate(brainAreas_all) if b'Amygdala' in area]
print(f"[ALL] Number of HC neurons: {len(hc_indices_all)}")
print(f"[ALL] Number of AMY neurons: {len(amy_indices_all)}")

# %%
# Step 2: Spike Count Matrix Creation for all_neurons using win_spike_rate, robust to multi-session
window_start = 0      # ms
window_end = 1000     # ms
n_trials_all = len(first_presentation_indices)
n_hc_all = len(hc_indices_all)
n_amy_all = len(amy_indices_all)
X_all = np.zeros((n_trials_all, n_hc_all))
Y_all = np.zeros((n_trials_all, n_amy_all))

for t, trial_idx in enumerate(first_presentation_indices):
    for i, neuron_idx in enumerate(hc_indices_all):
        neuron = all_neurons[neuron_idx]
        session_idx = neuron_session_idx[neuron_idx]
        # Use the correct trial object for this session and trial
        trial_obj = neuron.trials_recog[trial_idx]
        X_all[t, i] = trial_obj.win_spike_rate(neuron.spike_timestamps, window_start, window_end)
    for j, neuron_idx in enumerate(amy_indices_all):
        neuron = all_neurons[neuron_idx]
        session_idx = neuron_session_idx[neuron_idx]
        trial_obj = neuron.trials_recog[trial_idx]
        Y_all[t, j] = trial_obj.win_spike_rate(neuron.spike_timestamps, window_start, window_end)
print(f"[ALL] Spike count matrix X_all (HC): {X_all.shape}")
print(f"[ALL] Spike count matrix Y_all (AMY): {Y_all.shape}")

# %%
# Step 3: Data Preprocessing (Session-wise Z-score, Condition-specific PSTH, Residuals) for all_neurons

hc_session_idx = [neuron_session_idx[i] for i in hc_indices_all]
amy_session_idx = [neuron_session_idx[i] for i in amy_indices_all]

X_all_zscore = np.zeros_like(X_all)
Y_all_zscore = np.zeros_like(Y_all)
X_all_residual = np.zeros_like(X_all)
Y_all_residual = np.zeros_like(Y_all)

trial_types_all = trial_types  # Use pooled trial_types
unique_types_all = np.unique(trial_types_all)

for session_idx in range(len(nwb_files)):
    # Find neuron indices for this session
    hc_idx_this_sess = [i for i, s in enumerate(hc_session_idx) if s == session_idx]
    amy_idx_this_sess = [i for i, s in enumerate(amy_session_idx) if s == session_idx]
    # Z-score for this session's neurons
    if hc_idx_this_sess:
        X_sess = X_all[:, hc_idx_this_sess]
        X_sess_z = (X_sess - np.mean(X_sess, axis=0)) / (np.std(X_sess, axis=0) + 1e-10)
        X_all_zscore[:, hc_idx_this_sess] = X_sess_z
        n_neurons = X_sess_z.shape[1]
        n_conditions = len(unique_types_all)
        X_psths = np.zeros((n_conditions, n_neurons))
        for i, cond in enumerate(unique_types_all):
            mask = (trial_types_all == cond)
            X_psths[i, :] = np.mean(X_sess_z[mask], axis=0)
        for t in range(X_sess_z.shape[0]):
            cond_idx = np.where(unique_types_all == trial_types_all[t])[0][0]
            X_all_residual[t, hc_idx_this_sess] = X_sess_z[t, :] - X_psths[cond_idx, :]
    if amy_idx_this_sess:
        Y_sess = Y_all[:, amy_idx_this_sess]
        Y_sess_z = (Y_sess - np.mean(Y_sess, axis=0)) / (np.std(Y_sess, axis=0) + 1e-10)
        Y_all_zscore[:, amy_idx_this_sess] = Y_sess_z
        n_neurons = Y_sess_z.shape[1]
        n_conditions = len(unique_types_all)
        Y_psths = np.zeros((n_conditions, n_neurons))
        for i, cond in enumerate(unique_types_all):
            mask = (trial_types_all == cond)
            Y_psths[i, :] = np.mean(Y_sess_z[mask], axis=0)
        for t in range(Y_sess_z.shape[0]):
            cond_idx = np.where(unique_types_all == trial_types_all[t])[0][0]
            Y_all_residual[t, amy_idx_this_sess] = Y_sess_z[t, :] - Y_psths[cond_idx, :]

print('[ALL] Any NaNs in X_all_zscore?', np.isnan(X_all_zscore).any())
print('[ALL] Any NaNs in Y_all_zscore?', np.isnan(Y_all_zscore).any())
print(f"[ALL] X_all_residual shape: {X_all_residual.shape}")
print(f"[ALL] Y_all_residual shape: {Y_all_residual.shape}")

# %%
# Step 4: Communication Subspace Analysis (RRR, Cross-Validation) for all_neurons
from sklearn.model_selection import KFold
from numpy.linalg import pinv, svd

def reduced_rank_regression(Xc, Yc, rank):
    B_ols = pinv(Xc) @ Yc
    U, s, Vt = svd(B_ols, full_matrices=False)
    B_rrr = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
    return B_rrr

def cross_validate_rrr(X, Y, max_rank=10, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    ranks = range(1, min(max_rank, X.shape[1], Y.shape[1]) + 1)
    scores = {r: [] for r in ranks}
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        X_train_c = X_train - np.mean(X_train, axis=0)
        Y_train_c = Y_train - np.mean(Y_train, axis=0)
        X_test_c = X_test - np.mean(X_train, axis=0)
        Y_test_c = Y_test - np.mean(Y_train, axis=0)
        for r in ranks:
            B_rrr = reduced_rank_regression(X_train_c, Y_train_c, r)
            Y_pred = X_test_c @ B_rrr
            ss_res = np.sum((Y_test_c - Y_pred) ** 2)
            ss_tot = np.sum(Y_test_c ** 2)
            score = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            scores[r].append(score)
    avg_scores = {r: np.mean(scores[r]) for r in ranks}
    return avg_scores

max_rank_all = min(20, X_all_residual.shape[1], Y_all_residual.shape[1])
cv_scores_all = cross_validate_rrr(X_all_residual, Y_all_residual, max_rank=max_rank_all, n_splits=5)

plt.figure(figsize=(6,4))
ranks = list(cv_scores_all.keys())
plt.plot(ranks, [cv_scores_all[r] for r in ranks], marker='o')
plt.xlabel('Rank (Dimensionality)')
plt.ylabel('Cross-validated $R^2$')
plt.title('Prediction Performance vs. Rank (HC → AMY, all_neurons)')
plt.grid(True)
plt.show()

# --- AMY → HC analysis ---
max_rank_amy2hc = min(20, Y_all_residual.shape[1], X_all_residual.shape[1])
cv_scores_amy2hc = cross_validate_rrr(Y_all_residual, X_all_residual, max_rank=max_rank_amy2hc, n_splits=5)

plt.figure(figsize=(6,4))
ranks_amy2hc = list(cv_scores_amy2hc.keys())
plt.plot(ranks_amy2hc, [cv_scores_amy2hc[r] for r in ranks_amy2hc], marker='o', color='orange')
plt.xlabel('Rank (Dimensionality)')
plt.ylabel('Cross-validated $R^2$')
plt.title('Prediction Performance vs. Rank (AMY → HC, all_neurons)')
plt.grid(True)
plt.show()

# %%
# --- Step 5: Control Analyses (Within-area and Shuffle) for all_neurons ---

# Within-area: split HC neurons into two groups
split = X_all_residual.shape[1] // 2
X1 = X_all_residual[:, :split]
X2 = X_all_residual[:, split:]
within_scores = cross_validate_rrr(X1, X2, max_rank=min(50, X1.shape[1], X2.shape[1]), n_splits=5)

# Shuffle control: shuffle trial order for AMY
np.random.seed(42)
Y_shuffled = np.copy(Y_all_residual)
np.random.shuffle(Y_shuffled)
shuffle_scores = cross_validate_rrr(X_all_residual, Y_shuffled, max_rank=max_rank_all, n_splits=5)

# Plot all together
plt.figure(figsize=(7,5))
ranks = list(cv_scores_all.keys())
plt.plot(ranks, [cv_scores_all[r] for r in ranks], marker='o', label='HC → AMY (between-area)')
plt.plot(ranks[:len(within_scores)], [within_scores[r] for r in ranks[:len(within_scores)]], marker='s', label='HC split (within-area)')
plt.plot(ranks, [shuffle_scores[r] for r in ranks], marker='x', label='Shuffle control')
plt.xlabel('Rank (Dimensionality)')
plt.ylabel('Cross-validated $R^2$')
plt.title('Prediction Performance: Between, Within, Shuffle (all_neurons)')
plt.legend()
plt.grid(True)
plt.show()

# --- Step 5b: AMY → HC Control Analyses (Within-area and Shuffle) for all_neurons ---

# Within-area: split AMY neurons into two groups
split_amy = Y_all_residual.shape[1] // 2
Y1 = Y_all_residual[:, :split_amy]
Y2 = Y_all_residual[:, split_amy:]
within_scores_amy = cross_validate_rrr(Y1, Y2, max_rank=min(50, Y1.shape[1], Y2.shape[1]), n_splits=5)

# Shuffle control: shuffle trial order for HC
np.random.seed(42)
X_shuffled = np.copy(X_all_residual)
np.random.shuffle(X_shuffled)
shuffle_scores_amy = cross_validate_rrr(Y_all_residual, X_shuffled, max_rank=max_rank_amy2hc, n_splits=5)

# Plot all together for AMY → HC
plt.figure(figsize=(7,5))
ranks_amy2hc = list(cv_scores_amy2hc.keys())
plt.plot(ranks_amy2hc, [cv_scores_amy2hc[r] for r in ranks_amy2hc], marker='o', color='orange', label='AMY → HC (between-area)')
plt.plot(ranks_amy2hc[:len(within_scores_amy)], [within_scores_amy[r] for r in ranks_amy2hc[:len(within_scores_amy)]], marker='s', color='green', label='AMY split (within-area)')
plt.plot(ranks_amy2hc, [shuffle_scores_amy[r] for r in ranks_amy2hc], marker='x', color='red', label='Shuffle control')
plt.xlabel('Rank (Dimensionality)')
plt.ylabel('Cross-validated $R^2$')
plt.title('Prediction Performance: AMY → HC, Within, Shuffle (all_neurons)')
plt.legend()
plt.grid(True)
plt.show()

# %%
#inspect the Neuron(s)

#Get the spike times from the NWB file 

#Plot all the Neurons
for neuron in all_neurons:
    neuron.raster_psth(cell_type='visual', bin_size = 150)
# %%
