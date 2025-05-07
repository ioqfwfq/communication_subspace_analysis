import numpy as np
import matplotlib.pyplot as plt
import os

def plot_prediction_performance(ranks, cv_scores, title="Prediction Performance", save_path=None):
    """
    Plot prediction performance (R²) as a function of rank.
    
    Parameters:
    -----------
    ranks : list
        List of ranks
    cv_scores : dict
        Cross-validated R² for each rank
    title : str
        Plot title
    save_path : str
        Path to save the figure (if None, figure is not saved)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, [cv_scores[r] for r in ranks], 'o-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Cross-validated R²', fontsize=12)
    plt.title(title, fontsize=14)
    
    # Find optimal rank
    optimal_rank = max(cv_scores, key=cv_scores.get)
    plt.axvline(x=optimal_rank, color='r', linestyle='--', 
                label=f'Optimal rank: {optimal_rank} (R²={cv_scores[optimal_rank]:.3f})')
    plt.legend(fontsize=11)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_within_vs_between_area(ranks, within_scores, between_scores, title="Within vs. Between Area Prediction", save_path=None):
    """
    Compare prediction performance within an area vs. between areas.
    
    Parameters:
    -----------
    ranks : list
        List of ranks
    within_scores : dict
        Cross-validated R² for within-area prediction for each rank
    between_scores : dict
        Cross-validated R² for between-area prediction for each rank
    title : str
        Plot title
    save_path : str
        Path to save the figure (if None, figure is not saved)
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(ranks, [within_scores[r] for r in ranks], 'o-', linewidth=2, label='Within-area prediction')
    plt.plot(ranks, [between_scores[r] for r in ranks], 's-', linewidth=2, label='Between-area prediction')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Cross-validated R²', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    
    # Find optimal ranks
    optimal_within = max(within_scores, key=within_scores.get)
    optimal_between = max(between_scores, key=between_scores.get)
    
    plt.axvline(x=optimal_within, color='blue', linestyle='--', alpha=0.5)
    plt.axvline(x=optimal_between, color='orange', linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_individual_vs_pooled(individual_results, pooled_result, title="Individual vs. Pooled Analysis", save_path=None):
    """
    Compare results from individual subject analysis vs. pooled analysis.
    
    Parameters:
    -----------
    individual_results : list
        List of results from individual subjects
    pooled_result : dict
        Results from pooled analysis
    title : str
        Plot title
    save_path : str
        Path to save the figure (if None, figure is not saved)
    """
    # Extract optimal ranks and performance for each subject
    subject_ids = range(len(individual_results))
    ranks = [result['optimal_rank'] for result in individual_results]
    performance = [max(result['between_area_scores'].values()) for result in individual_results]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot optimal ranks
    ax1.bar(subject_ids, ranks, color='skyblue', alpha=0.7)
    ax1.axhline(y=pooled_result['optimal_rank'], color='r', linestyle='--', 
               label=f'Pooled rank: {pooled_result["optimal_rank"]}')
    ax1.set_xlabel('Subject ID', fontsize=12)
    ax1.set_ylabel('Optimal Rank', fontsize=12)
    ax1.set_title('Communication Subspace Dimensionality by Subject', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot performance
    ax2.bar(subject_ids, performance, color='lightgreen', alpha=0.7)
    ax2.axhline(y=pooled_result['pooled_performance'], color='r', linestyle='--',
               label=f'Pooled performance: {pooled_result["pooled_performance"]:.3f}')
    ax2.set_xlabel('Subject ID', fontsize=12)
    ax2.set_ylabel('Prediction Performance (R²)', fontsize=12)
    ax2.set_title('Communication Subspace Performance by Subject', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_validation_scores(validation_scores, title="Pooled Subspace Validation", save_path=None):
    """
    Plot validation scores of pooled subspace on individual subjects.
    
    Parameters:
    -----------
    validation_scores : list
        Validation scores for each subject
    title : str
        Plot title
    save_path : str
        Path to save the figure (if None, figure is not saved)
    """
    plt.figure(figsize=(10, 6))
    
    subject_ids = range(len(validation_scores))
    plt.bar(subject_ids, validation_scores, color='purple', alpha=0.7)
    
    plt.axhline(y=np.mean(validation_scores), color='r', linestyle='--',
               label=f'Mean: {np.mean(validation_scores):.3f}')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Subject ID', fontsize=12)
    plt.ylabel('Validation Score (R²)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()