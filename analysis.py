import os
import numpy as np
import matplotlib.pyplot as plt
from scripts.data_processing import extract_spikes_from_nwb, create_spike_count_matrices, preprocess_subject_data
from scripts.rrr import reduced_rank_regression, cross_validate_rrr, get_communication_subspace, subspace_alignment
from scripts.visualization import (plot_prediction_performance, plot_within_vs_between_area, 
                                plot_individual_vs_pooled, plot_validation_scores)
from sklearn.decomposition import PCA
import glob

# Create output directory for figures
if not os.path.exists('figures'):
    os.makedirs('figures')

def analyze_individual_subject(file_path, region1='IT', region2='HC'):
    """
    Analyze one subject to find their communication subspace.
    
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
    results : dict
        Results of the analysis
    """
    print(f"\nAnalyzing subject: {os.path.basename(file_path)}")
    
    # Extract data
    spike_times, trial_starts, trial_ends, trial_types, region1_indices, region2_indices = extract_spikes_from_nwb(file_path, region1, region2)
    
    # Create spike count matrices
    X, Y = create_spike_count_matrices(spike_times, trial_starts, trial_ends, region1_indices, region2_indices, region1, region2)
    
    # Preprocess data
    X_zscore, Y_zscore, X_residual, Y_residual = preprocess_subject_data(X, Y, trial_types)
    
    # Find intrinsic dimensionality of each area using PCA
    pca_x = PCA().fit(X_residual)
    pca_y = PCA().fit(Y_residual)
    
    # Find communication subspace using RRR
    ranks = range(1, min(X_residual.shape[1], Y_residual.shape[1]) + 1)
    between_area_scores = cross_validate_rrr(X_residual, Y_residual, ranks)
    
    # Find optimal rank
    optimal_rank = max(between_area_scores, key=between_area_scores.get)
    
    # Get communication subspace at optimal rank
    communication_subspace = get_communication_subspace(X_residual, Y_residual, optimal_rank)
    
    # Get within-area prediction performance
    # Split X into two equal parts
    n_neurons_x = X_residual.shape[1]
    split_idx = n_neurons_x // 2
    X1 = X_residual[:, :split_idx]
    X2 = X_residual[:, split_idx:]
    
    # Predict X2 from X1
    within_area_scores = cross_validate_rrr(X1, X2, ranks)
    
    # Plot results
    subject_name = os.path.basename(file_path).split('.')[0]
    
    plot_prediction_performance(
        ranks, 
        between_area_scores, 
        title=f"Between-area prediction ({region1} → {region2}): {subject_name}",
        save_path=f"figures/{subject_name}_between_area_prediction.png"
    )
    
    plot_within_vs_between_area(
        ranks, 
        within_area_scores, 
        between_area_scores, 
        title=f"Within vs. Between Area Prediction: {subject_name}",
        save_path=f"figures/{subject_name}_within_vs_between.png"
    )
    
    results = {
        'X_residual': X_residual,
        'Y_residual': Y_residual,
        'X_var_explained': np.cumsum(pca_x.explained_variance_ratio_),
        'Y_var_explained': np.cumsum(pca_y.explained_variance_ratio_),
        'between_area_scores': between_area_scores,
        'within_area_scores': within_area_scores,
        'optimal_rank': optimal_rank,
        'communication_subspace': communication_subspace
    }
    
    return results

def align_subspaces(subspaces):
    """
    Align communication subspaces across subjects using Procrustes analysis.
    
    Parameters:
    -----------
    subspaces : list
        List of communication subspaces from different subjects
        
    Returns:
    --------
    aligned_subspaces : list
        List of aligned communication subspaces
    """
    # This is a simplified implementation
    # A more sophisticated approach would use Procrustes analysis
    # or canonical correlation analysis
    
    # For now, just ensure all subspaces have the same dimensionality
    min_dim = min([subspace.shape[1] for subspace in subspaces])
    
    aligned_subspaces = [subspace[:, :min_dim] for subspace in subspaces]
    
    return aligned_subspaces

def pool_data_across_subjects(subject_results, region1='IT', region2='HC'):
    """
    Pool preprocessed data across subjects.
    
    Parameters:
    -----------
    subject_results : list
        List of results from individual subjects
    region1 : str
        Name of the first brain region
    region2 : str
        Name of the second brain region
        
    Returns:
    --------
    X_pooled : numpy.ndarray
        Pooled source area activity
    Y_pooled : numpy.ndarray
        Pooled target area activity
    """
    # Concatenate X and Y matrices across subjects
    X_pooled = np.vstack([result['X_residual'] for result in subject_results])
    Y_pooled = np.vstack([result['Y_residual'] for result in subject_results])
    
    print(f"Pooled data dimensions:")
    print(f"{region1}: {X_pooled.shape} (trials × neurons)")
    print(f"{region2}: {Y_pooled.shape} (trials × neurons)")
    
    return X_pooled, Y_pooled

def validate_pooled_subspace(pooled_subspace, subject_results):
    """
    Validate pooled communication subspace on individual subjects.
    
    Parameters:
    -----------
    pooled_subspace : numpy.ndarray
        Pooled communication subspace
    subject_results : list
        List of results from individual subjects
        
    Returns:
    --------
    validation_scores : list
        Validation scores for each subject
    """
    validation_scores = []
    
    for result in subject_results:
        X = result['X_residual']
        Y = result['Y_residual']
        
        # Predict Y using pooled subspace
        # First project X onto the pooled subspace
        X_centered = X - np.mean(X, axis=0)
        Y_centered = Y - np.mean(Y, axis=0)
        
        # Create regression weights using the pooled subspace dimensions
        W = pooled_subspace @ pooled_subspace.T @ np.linalg.pinv(X_centered) @ Y_centered
        
        # Predict Y
        Y_pred = X_centered @ W
        
        # Calculate explained variance
        score = 1 - np.sum((Y_centered - Y_pred)**2) / np.sum(Y_centered**2)
        validation_scores.append(score)
    
    return validation_scores

def run_hierarchical_analysis(nwb_file_paths, region1='IT', region2='HC'):
    """
    Run the full hierarchical communication subspace analysis.
    
    Parameters:
    -----------
    nwb_file_paths : list
        List of paths to NWB files
    region1 : str
        Name of the first brain region
    region2 : str
        Name of the second brain region
        
    Returns:
    --------
    results : dict
        Results of the analysis
    """
    print(f"Running hierarchical analysis between {region1} and {region2}")
    print(f"Found {len(nwb_file_paths)} subjects")
    
    # Step 1: Individual subject analysis
    subject_results = []
    for file_path in nwb_file_paths:
        try:
            result = analyze_individual_subject(file_path, region1, region2)
            subject_results.append(result)
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            continue
    
    print(f"Successfully analyzed {len(subject_results)} subjects")
    
    if len(subject_results) == 0:
        print("No subjects could be analyzed. Exiting.")
        return None
    
    # Step 2: Extract and align communication subspaces
    communication_subspaces = [result['communication_subspace'] for result in subject_results]
    aligned_subspaces = align_subspaces(communication_subspaces)
    
    # Step 3: Pooled analysis
    X_pooled, Y_pooled = pool_data_across_subjects(subject_results, region1, region2)
    ranks = range(1, min(X_pooled.shape[1], Y_pooled.shape[1]) + 1)
    pooled_cv_scores = cross_validate_rrr(X_pooled, Y_pooled, ranks)
    
    # Plot pooled analysis results
    plot_prediction_performance(
        ranks, 
        pooled_cv_scores, 
        title=f"Pooled Analysis: {region1} → {region2}",
        save_path=f"figures/pooled_prediction_performance.png"
    )
    
    # Find optimal rank for pooled data
    pooled_optimal_rank = max(pooled_cv_scores, key=pooled_cv_scores.get)
    
    # Get pooled communication subspace
    pooled_subspace = get_communication_subspace(X_pooled, Y_pooled, pooled_optimal_rank)
    
    # Step 4: Validate pooled subspace on individual subjects
    validation_scores = validate_pooled_subspace(pooled_subspace, subject_results)
    
    # Plot validation results
    plot_validation_scores(
        validation_scores, 
        title=f"Validation of Pooled Subspace on Individual Subjects",
        save_path=f"figures/pooled_validation_scores.png"
    )
    
    # Step 5: Compare individual vs. pooled results
    pooled_result = {
        'optimal_rank': pooled_optimal_rank,
        'pooled_performance': max(pooled_cv_scores.values())
    }
    
    plot_individual_vs_pooled(
        subject_results, 
        pooled_result, 
        title=f"Individual vs. Pooled Analysis: {region1} → {region2}",
        save_path=f"figures/individual_vs_pooled.png"
    )
    
    # Step 6: Compile and return results
    results = {
        'individual_results': subject_results,
        'individual_optimal_ranks': [result['optimal_rank'] for result in subject_results],
        'individual_performance': [max(result['between_area_scores'].values()) for result in subject_results],
        'pooled_optimal_rank': pooled_optimal_rank,
        'pooled_performance': max(pooled_cv_scores.values()),
        'validation_scores': validation_scores,
        'pooled_subspace': pooled_subspace
    }
    
    return results

if __name__ == "__main__":
    # Set regions to analyze
    region1 = 'IT'   # Source region
    region2 = 'HC'   # Target region
    
    # Find all NWB files in the data directory
    nwb_files = glob.glob("data/*.nwb")
    
    if len(nwb_files) == 0:
        print("No NWB files found in the data directory.")
        print("Please place your NWB files in the 'data' folder.")
    else:
        # Run the analysis
        results = run_hierarchical_analysis(nwb_files, region1, region2)
        
        if results:
            print("\nAnalysis complete!")
            print(f"Results saved in the 'figures' directory")