# %% [markdown]
# # Introduction 
# 
# The purpose of this Jupyter Notebook is to illustrate key processes in our software pipeline that utlilizes NWB. Our hope is that this notebook can serve as a tutorial on reading and extracting data from an NWB file. 
# 
# * Author: [Nand Chandravadia](mailto:nandc10@ucla.edu)
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
# Input Path to the NWB file here:
pathtoNWBFile = 'data/P16HMH_NOID24.nwb'
nwbBasePath = Path(pathtoNWBFile)
# %%
#Read the nwb file 
io = NWBHDF5IO(str(nwbBasePath), mode='r')
nwb = io.read()

#Get the fields within the NWB file
nwbFields = nwb.fields
print('These are the top-level Groups within the NWB file: {}\n'.format(nwbFields.keys()))

#Get Meta-Data from NWB file 
print('The experiment within this NWB file was conducted at {} in the lab of {}. The experiment is detailed as follows: {}'.format(nwb.institution, nwb.lab, nwb.experiment_description))
# %% [markdown]
# ## Read Data and Meta-data
# Here, we read the data and meta-data from the specified NWB file using the NWB read utility. 
# 
# The NWB file is composed of various Groups, Datasets, and Attributes. The data and cooresponding meta-data are encapsulated within these Groups. The data are thus organized according to these Groups. We can also read the data and meta-data within these Groups, and visualize the components within NWB file via the *nwb2widget* utility -- the following illustrates this process:
# %%
##Plot the Waveforms from the NWB file

#Which channel_index to plot? 
channel_index = [0]

# get Waveform Means from the NWB file
allwaveformLearn = np.asarray(nwb.units['waveform_mean_encoding'].data)
allwaveformRecog = np.asarray(nwb.units['waveform_mean_recognition'].data)

# Choose Which Channel Index to Plot
waveformLearn = allwaveformLearn[channel_index, :][0]
waveformRecog = allwaveformLearn[channel_index, :][0]

#get brain Areas
brainAreas = np.asarray(nwb.electrodes['location'].data)


#Plot the mean waveforms

fig, axes = plt.subplots(1, 2, figsize = (15, 10)) 

#Plot Learning
axes[0].plot(range(len(waveformLearn)), waveformLearn, color = 'blue', marker = 'o', linestyle='dashed',
            linewidth=1, markersize=3)
axes[0].set_title('Learning, session: {}, brain Area: {}'.format(nwb.identifier, brainAreas[channel_index][0]))
axes[0].set_xlabel('time (in ms)')
axes[0].set_ylabel('\u03BCV')


#Plot Recog
axes[1].plot(range(len(waveformRecog)), waveformRecog, color = 'green', marker = 'o', linestyle='dashed',
            linewidth=1, markersize=3)
axes[1].set_title('Recognition, session: {}, brain Area: {}'.format(nwb.identifier, brainAreas[channel_index][0]))
axes[1].set_xlabel('time (in ms)')
axes[1].set_ylabel('\u03BCV')


# %% [markdown]
# ## Extracting and Plotting the Mean Waveform(s)
# 
# To extract the mean waveform, we simply call waveform_mean_encoding from the \units table -- *nwb.units['waveform_mean_encoding']*. The brain area of each of the electrodes is located within the \electrodes table -- *nwb.electrodes['location']*. To see the relationship bewteen the \units and \electrodes table, see **Figure 2b** in our data descriptor. 
# %%
#Plot Behavior from NWB file 

# == Plot ROC curve == 
fig, axes = plt.subplots(1, 3, figsize = (25, 10))
# Calculate the cumulative d and plot the cumulative ROC curve
stats_all = helper.cal_cumulative_d(nwb)
# Calculate the auc
auc = helper.cal_auc(stats_all)
x = stats_all[0:5, 4]
y = stats_all[0:5, 3]
axes[2].plot(x, y, marker='.', color='grey', alpha=0.5, linewidth=2, markersize=3)
axes[2].set_ylim(0, 1)
axes[2].set_xlim(0, 1)
axes[2].set_title('ROC Curve, AUC: {}'.format(auc))
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].plot([0, 1], [0, 1], color='black', alpha=0.5, linewidth=2)


#Get the recognition responses
recog_response = helper.extract_recog_responses(nwb)
ground_truth = helper.extract_new_old_label(nwb)
#Get the recognition responses for the 'old' stimuli
recog_response_old = recog_response[ground_truth == 1]

# Place holder ready to store separate the new and old response
response_1_old = []
response_2_old = []
response_3_old = []
response_4_old = []
response_5_old = []
response_6_old = []

response_1_new = []
response_2_new = []
response_3_new = []
response_4_new = []
response_5_new = []
response_6_new = []


# Calculate the percentage of each responses
response_1_old.append(np.sum(recog_response_old == 1) / len(recog_response_old))
response_2_old.append(np.sum(recog_response_old == 2) / len(recog_response_old))
response_3_old.append(np.sum(recog_response_old == 3) / len(recog_response_old))
response_4_old.append(np.sum(recog_response_old == 4) / len(recog_response_old))
response_5_old.append(np.sum(recog_response_old == 5) / len(recog_response_old))
response_6_old.append(np.sum(recog_response_old == 6) / len(recog_response_old))

recog_response_new = recog_response[ground_truth == 0]
response_1_new.append(np.sum(recog_response_new == 1) / len(recog_response_new))
response_2_new.append(np.sum(recog_response_new == 2) / len(recog_response_new))
response_3_new.append(np.sum(recog_response_new == 3) / len(recog_response_new))
response_4_new.append(np.sum(recog_response_new == 4) / len(recog_response_new))
response_5_new.append(np.sum(recog_response_new == 5) / len(recog_response_new))
response_6_new.append(np.sum(recog_response_new == 6) / len(recog_response_new))


# Plot the percentage responses
response_old = np.asarray([response_1_old, response_2_old, response_3_old, response_4_old,
                               response_5_old, response_6_old])
response_new = np.asarray([response_1_new, response_2_new, response_3_new, response_4_new,
                               response_5_new, response_6_new])

n = 1
response_percentage_old = np.mean(response_old, axis=1)
std_old = np.std(response_old)
se_old = std_old/np.sqrt(n)
response_percentage_new = np.mean(response_new, axis=1)
std_new = np.std(response_new)
se_new = std_new/np.sqrt(n)

#x = [i for i in range(1, 7, 1)]
x = ['New, very sure', 'New, sure', 'New, unsure', 'Old, unsure', 'Old, sure', 'Old, very sure']
axes[1].errorbar(x, response_percentage_old, yerr=se_old, color='blue', label='old stimuli', linewidth = 2)
axes[1].errorbar(x, response_percentage_new, yerr=se_new, color='red', label='new stimuli', linewidth = 2)
axes[1].legend()
axes[1].set_xlabel('Confidence')
axes[1].set_ylabel('Probability of Response')



# == Plot the Response Times for correct vs. incorrect responses == 

events_learn, timestamps_learn, events_recog, timestamps_recog = helper.get_event_data(nwb)

data_events = {'events_recog': events_recog, 'timestamps_recog': timestamps_recog}
recog = pd.DataFrame(data_events) 

index_questionScreenOnset = list(np.where(recog['events_recog'] == 3)) #question screen onset

response_recog = helper.extract_recog_responses(nwb)
ground_truth = helper.extract_new_old_label(nwb)
correct_ind, incorrect_ind = helper.correct_incorrect_indexes(recog_response, ground_truth)

responseTimesRecogCorrect = []
responseTimesRecogIncorrect = []

#Get response times for correct trials, from question screen onset
for a in index_questionScreenOnset[0][correct_ind]: 
    responseTime = recog.iloc[a+1, 1] - recog.iloc[a, 1]
    responseTimesRecogCorrect.append(responseTime)
    
#Get response times for incorrect trials, from question screen onset
for a in index_questionScreenOnset[0][incorrect_ind]: 
    responseTime = recog.iloc[a+1, 1] - recog.iloc[a, 1]
    responseTimesRecogIncorrect.append(responseTime)

accuracy = len(correct_ind)/len(response_recog)


responseTimesAll = responseTimesRecogCorrect + responseTimesRecogIncorrect
trial_indication = ['Correct']*len(responseTimesRecogCorrect) + ['Incorrect']*len(responseTimesRecogIncorrect)
dict_responseTimes = {'time': responseTimesAll, 'trial': trial_indication}
dataframe_responseTimes = pd.DataFrame(dict_responseTimes)

#Plot Boxplot
sns.boxplot(x = 'trial', y= "time", data = dataframe_responseTimes, ax = axes[0], color = 'b')
sns.swarmplot(x = 'trial', y= "time", data=dataframe_responseTimes, color="g", ax = axes[0])
axes[0].set_ylabel('time (measured in s)')
axes[0].set_title('Response times for all trials, accuracy: {}'.format(accuracy))

# %% [markdown]
# ## Behavior
# 
# We can plot the behavior from the NWB file. The behavioral data is mostly encapsulated within nwb\trials, which includes the trial information such as the start_time and response_time. 
# %%
#Plot the Neuron(s)

#Get the spike times from the NWB file 
index = 0
nwb.units.get_unit_spike_times(index)

neurons = single_neuron.extract_neuron_data_from_nwb(nwb)

#Plot all the Neurons
for neuron in neurons:
    neuron.raster_psth(cell_type='visual', bin_size = 150)
    neuron.raster_psth(cell_type = 'memory', bin_size = 150)

# %% [markdown]
# ## Single Neuron Analysis
# 
# Here, we demonstrate how to run single neuron analysis in NWB. We probed for the tuning of two functional cell types, MS and VS cells, Memory Selective, and  Visually Seletive, respectively.


# %% [markdown]
# ## Exploring NWB File Structure
# 
# Before extracting specific data, we need to understand how neuron locations and trial types are stored in the NWB file.  
# In this step, we will:
# - Load the NWB file
# - Print the available fields in `units`, `electrodes`, and `trials`
# - Display a few example entries from each to guide further analysis

# %%
# Path to your NWB file (update if needed)
# Print top-level fields
print("Top-level NWB fields:", list(nwb.fields.keys()))

# Explore units (neurons)
print("\n--- UNITS TABLE ---")
print("Units fields:", list(nwb.units.colnames))
for col in nwb.units.colnames:
    data = nwb.units[col].data[:5] if hasattr(nwb.units[col], 'data') else nwb.units[col][:5]
    print(f"{col}: {data}")

# Explore electrodes (for neuron locations)
print("\n--- ELECTRODES TABLE ---")
print("Electrodes fields:", list(nwb.electrodes.colnames))
for col in nwb.electrodes.colnames:
    data = nwb.electrodes[col].data[:5] if hasattr(nwb.electrodes[col], 'data') else nwb.electrodes[col][:5]
    print(f"{col}: {data}")

# Explore trials (for trial types)
print("\n--- TRIALS TABLE ---")
print("Trials fields:", list(nwb.trials.colnames))
for col in nwb.trials.colnames:
    data = nwb.trials[col].data[:5] if hasattr(nwb.trials[col], 'data') else nwb.trials[col][:5]
    print(f"{col}: {data}")

# Close the file
# io.close()

# %%
# Get all neuron indices
n_neurons = len(nwb.units['electrodes'].data)
neuron_electrode_indices = nwb.units['electrodes'].data[:]
electrode_locations = nwb.electrodes['location'].data[:]

# Filter out units with out-of-bounds electrode indices
valid_unit_indices = [i for i, elec_idx in enumerate(neuron_electrode_indices)
                      if elec_idx < len(electrode_locations)]

# Find hippocampus and amygdala neuron indices using only valid units
hc_neuron_indices = [i for i in valid_unit_indices
                     if b'hippocampus' in electrode_locations[neuron_electrode_indices[i]].lower()]
amy_neuron_indices = [i for i in valid_unit_indices
                      if b'amygdala' in electrode_locations[neuron_electrode_indices[i]].lower()]

print(f"Hippocampus neuron indices: {hc_neuron_indices}")
print(f"Amygdala neuron indices: {amy_neuron_indices}")

# --- Trials: Find first-presentation (encoding) trials ---
stim_phase = nwb.trials['stim_phase'].data[:]
new_old_labels = nwb.trials['new_old_labels_recog'].data[:]

# Encoding trials: stim_phase == b'learn'
encoding_trial_indices = [i for i, phase in enumerate(stim_phase) if phase == b'learn']

# Optionally, filter for trials where new_old_labels_recog is b'NA' (first presentation)
first_presentation_indices = [i for i in encoding_trial_indices if new_old_labels[i] == b'NA']

print(f"First-presentation (encoding) trial indices: {first_presentation_indices}")

# %%
# --- Step 1: Data Extraction for Communication Subspace Analysis ---
# Using existing brainAreas variable from above


# Find hippocampus and amygdala neurons among valid indices
hc_neuron_indices = [i for i, area in enumerate(brainAreas)
                     if b'Hippocampus' in area]
amy_neuron_indices = [i for i, area in enumerate(brainAreas)
                      if b'Amygdala' in area]

print(f"Number of HC neurons: {len(hc_neuron_indices)}")
print(f"Number of AMY neurons: {len(amy_neuron_indices)}")

# Get the concatenated spike times and the index array
all_spike_times = nwb.units['spike_times'].data[:]
spike_times_index = nwb.units['spike_times_index'].data[:]

# For each unit, extract its spike times
unit_spike_times = []
start = 0
for end in spike_times_index:
    unit_spike_times.append(all_spike_times[start:end])
    start = end

# Extract trial information
trial_starts = nwb.trials['start_time'].data[:]
trial_ends = nwb.trials['stop_time'].data[:]
stim_phase = nwb.trials['stim_phase'].data[:]
new_old_labels = nwb.trials['new_old_labels_recog'].data[:]

# Filter for encoding (first-presentation) trials: stim_phase == b'learn' and new_old_labels == b'NA'
encoding_trial_indices = [i for i, phase in enumerate(stim_phase) if phase == b'learn']
first_presentation_indices = [i for i in encoding_trial_indices if new_old_labels[i] == b'NA']

print(f"Number of first-presentation (encoding) trials: {len(first_presentation_indices)}")

# Store trial start times for first-presentation trials
encoding_trial_starts = trial_starts[first_presentation_indices]
encoding_trial_ends = trial_ends[first_presentation_indices]

# %%
# --- Step 2: Spike Count Matrix Creation ---

# Define time window (in seconds) after stimulus onset
window_start = 0.0
window_end = 1.0  # 0-1s (1000 ms)

# Check if we have any valid neurons before getting counts
n_trials = len(first_presentation_indices)
n_hc = len(hc_neuron_indices) if hc_neuron_indices else 0
n_amy = len(amy_neuron_indices) if amy_neuron_indices else 0

# Initialize spike count matrices: (trials, neurons)
X = np.zeros((n_trials, n_hc))  # HC
Y = np.zeros((n_trials, n_amy))  # AMY

for t, trial_idx in enumerate(first_presentation_indices):
    t0 = trial_starts[trial_idx] + window_start
    t1 = trial_starts[trial_idx] + window_end
    # HC neurons
    for i, neuron_idx in enumerate(hc_neuron_indices):
        spikes = neurons[neuron_idx].spike_timestamps / 1e6  # convert to seconds
        X[t, i] = np.sum((spikes >= t0) & (spikes < t1))
    # AMY neurons
    for j, neuron_idx in enumerate(amy_neuron_indices):
        spikes = neurons[neuron_idx].spike_timestamps / 1e6  # convert to seconds
        Y[t, j] = np.sum((spikes >= t0) & (spikes < t1))

print(f"Spike count matrix X (HC): {X.shape}")
print(f"Spike count matrix Y (AMY): {Y.shape}")

# --- Step 3: Data Preprocessing (Session-wise Z-score, Condition-specific PSTH, Residuals) for all_neurons

hc_session_idx = [neuron_session_idx[i] for i in hc_neuron_indices]
amy_session_idx = [neuron_session_idx[i] for i in amy_neuron_indices]

# Z-score each neuron's activity across trials
def zscore_matrix(mat):
    return (mat - np.mean(mat, axis=0)) / (np.std(mat, axis=0) + 1e-10)

X_zscore = zscore_matrix(X)
Y_zscore = zscore_matrix(Y)

print('Any NaNs in X_zscore?', np.isnan(X_zscore).any())
print('Any NaNs in Y_zscore?', np.isnan(Y_zscore).any())

# Try to find a trial-type label for encoding trials (e.g., stimulus category)
# If not available, treat all as one type
trial_type_col = 'stimCategory'
# for col in nwb.trials.colnames:
#     if col not in ['start_time', 'stop_time', 'stim_phase', 'new_old_labels_recog']:
#         trial_type_col = col
#         break
print(nwb.trials.colnames)
if trial_type_col:
    trial_types = nwb.trials[trial_type_col].data[first_presentation_indices]
    unique_types = np.unique(trial_types)
else:
    trial_types = np.zeros(len(first_presentation_indices))
    unique_types = [0]

print('trial_type_col:', trial_type_col)
print('trial_types (first 10):', trial_types[:10])
print('unique_types:', unique_types)

# Compute PSTH for each trial type and subtract to get residuals
X_residual = np.zeros_like(X_zscore)
Y_residual = np.zeros_like(Y_zscore)

for ttype in unique_types:
    mask = (trial_types == ttype)
    X_psth = np.mean(X_zscore[mask], axis=0)
    Y_psth = np.mean(Y_zscore[mask], axis=0)
    X_residual[mask] = X_zscore[mask] - X_psth
    Y_residual[mask] = Y_zscore[mask] - Y_psth

print(f"X_residual shape: {X_residual.shape}")
print(f"Y_residual shape: {Y_residual.shape}")

# %%
# --- Step 4: Communication Subspace Analysis (RRR, Cross-Validation) ---

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Helper: RRR via SVD
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

# Run cross-validated RRR
max_rank = min(10, X_residual.shape[1], Y_residual.shape[1])
cv_scores = cross_validate_rrr(Y_residual, X_residual, max_rank=max_rank, n_splits=5)

# Plot prediction performance vs. rank
plt.figure(figsize=(6,4))
ranks = list(cv_scores.keys())
plt.plot(ranks, [cv_scores[r] for r in ranks], marker='o')
plt.xlabel('Rank (Dimensionality)')
plt.ylabel('Cross-validated $R^2$')
plt.title('Prediction Performance vs. Rank (HC → AMY)')
plt.grid(True)
plt.show()

# %%
# --- Step 5: Control Analyses (Within-area and Shuffle) ---

# Within-area: split HC neurons into two groups
split = X_residual.shape[1] // 2
X1 = X_residual[:, :split]
X2 = X_residual[:, split:]
within_scores = cross_validate_rrr(X1, X2, max_rank=min(10, X1.shape[1], X2.shape[1]), n_splits=5)

# Shuffle control: shuffle trial order for AMY
np.random.seed(42)
Y_shuffled = np.copy(Y_residual)
np.random.shuffle(Y_shuffled)
shuffle_scores = cross_validate_rrr(X_residual, Y_shuffled, max_rank=max_rank, n_splits=5)

# Plot all together
plt.figure(figsize=(7,5))
ranks = list(cv_scores.keys())
plt.plot(ranks, [cv_scores[r] for r in ranks], marker='o', label='HC → AMY (between-area)')
plt.plot(ranks[:len(within_scores)], [within_scores[r] for r in ranks[:len(within_scores)]], marker='s', label='HC split (within-area)')
plt.plot(ranks, [shuffle_scores[r] for r in ranks], marker='x', label='Shuffle control')
plt.xlabel('Rank (Dimensionality)')
plt.ylabel('Cross-validated $R^2$')
plt.title('Prediction Performance: Between, Within, Shuffle')
plt.legend()
plt.grid(True)
plt.show()

# %%
# --- Step 6: Visualization and Interpretation ---

# Find optimal rank (where between-area performance saturates)
optimal_rank = max(cv_scores, key=cv_scores.get)
print(f"Optimal rank: {optimal_rank}")

# Get communication subspace weights (HC → AMY)
B_rrr_opt = reduced_rank_regression(X_residual, Y_residual, optimal_rank)

# Plot weights for first communication dimension (HC neurons)
plt.figure(figsize=(8,3))
plt.bar(range(X_residual.shape[1]), B_rrr_opt[:,0])
plt.xlabel('HC neuron index')
plt.ylabel('Weight (1st comm. dim)')
plt.title('Top Communication Dimension Weights (HC → AMY)')
plt.show()

# Project data onto first communication dimension
proj = X_residual @ B_rrr_opt[:,0]
plt.figure(figsize=(6,3))
plt.hist(proj, bins=20, color='purple', alpha=0.7)
plt.xlabel('Projection onto 1st comm. dim')
plt.ylabel('Number of trials')
plt.title('Trial Projections onto 1st Communication Dimension')
plt.show()

hc_spike_times = [unit_spike_times[i] for i in hc_neuron_indices]
amy_spike_times = [unit_spike_times[i] for i in amy_neuron_indices]

# %%
# --- all neuron analysis Load all NWB files and merge all neuron objects into all_neurons
nwb_dir = Path('data')
nwb_files = sorted(list(nwb_dir.glob('*.nwb')))
print(f"[END] Found {len(nwb_files)} NWB files:", nwb_files)

all_neurons = []
brainAreas_all = []
neuron_session_idx = []  # Track session index for each neuron
for session_idx, nwb_path in enumerate(nwb_files[13:14]):
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
# Step 2: Spike Count Matrix Creation for all_neurons
window_start = 0.0
window_end = 1.0
n_trials_all = len(first_presentation_indices)
n_hc_all = len(hc_indices_all)
n_amy_all = len(amy_indices_all)
X_all = np.zeros((n_trials_all, n_hc_all))
Y_all = np.zeros((n_trials_all, n_amy_all))
for t, trial_idx in enumerate(first_presentation_indices):
    t0 = trial_starts[trial_idx] + window_start
    t1 = trial_starts[trial_idx] + window_end
    for i, neuron_idx in enumerate(hc_indices_all):
        spikes = all_neurons[neuron_idx].spike_timestamps / 1e6
        X_all[t, i] = np.sum((spikes >= t0) & (spikes < t1))
    for j, neuron_idx in enumerate(amy_indices_all):
        spikes = all_neurons[neuron_idx].spike_timestamps / 1e6
        Y_all[t, j] = np.sum((spikes >= t0) & (spikes < t1))
print(f"[ALL] Spike count matrix X_all (HC): {X_all.shape}")
print(f"[ALL] Spike count matrix Y_all (AMY): {Y_all.shape}")

# %%
# Step 3: Data Preprocessing (Session-wise Z-score, PSTH, Residuals) for all_neurons

hc_session_idx = [neuron_session_idx[i] for i in hc_indices_all]
amy_session_idx = [neuron_session_idx[i] for i in amy_indices_all]

X_all_zscore = np.zeros_like(X_all)
Y_all_zscore = np.zeros_like(Y_all)
X_all_residual = np.zeros_like(X_all)
Y_all_residual = np.zeros_like(Y_all)

for session_idx in range(len(nwb_files)):
    # Find neuron indices for this session
    hc_idx_this_sess = [i for i, s in enumerate(hc_session_idx) if s == session_idx]
    amy_idx_this_sess = [i for i, s in enumerate(amy_session_idx) if s == session_idx]
    if hc_idx_this_sess:
        X_sess = X_all[:, hc_idx_this_sess]
        X_sess_z = (X_sess - np.mean(X_sess, axis=0)) / (np.std(X_sess, axis=0) + 1e-10)
        X_all_zscore[:, hc_idx_this_sess] = X_sess_z
    if amy_idx_this_sess:
        Y_sess = Y_all[:, amy_idx_this_sess]
        Y_sess_z = (Y_sess - np.mean(Y_sess, axis=0)) / (np.std(Y_sess, axis=0) + 1e-10)
        Y_all_zscore[:, amy_idx_this_sess] = Y_sess_z

print('[ALL] Any NaNs in X_all_zscore?', np.isnan(X_all_zscore).any())
print('[ALL] Any NaNs in Y_all_zscore?', np.isnan(Y_all_zscore).any())

trial_types_all = trial_types  # Use pooled trial_types
unique_types_all = np.unique(trial_types_all)

for ttype in unique_types_all:
    mask = (trial_types_all == ttype)
    for session_idx in range(len(nwb_files)):
        hc_idx_this_sess = [i for i, s in enumerate(hc_session_idx) if s == session_idx]
        amy_idx_this_sess = [i for i, s in enumerate(amy_session_idx) if s == session_idx]
        if hc_idx_this_sess:
            X_psth = np.mean(X_all_zscore[mask][:, hc_idx_this_sess], axis=0)
            X_all_residual[mask][:, hc_idx_this_sess] = X_all_zscore[mask][:, hc_idx_this_sess] - X_psth
        if amy_idx_this_sess:
            Y_psth = np.mean(Y_all_zscore[mask][:, amy_idx_this_sess], axis=0)
            Y_all_residual[mask][:, amy_idx_this_sess] = Y_all_zscore[mask][:, amy_idx_this_sess] - Y_psth

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

max_rank_all = min(50, X_all_residual.shape[1], Y_all_residual.shape[1])
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
max_rank_amy2hc = min(50, Y_all_residual.shape[1], X_all_residual.shape[1])
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
