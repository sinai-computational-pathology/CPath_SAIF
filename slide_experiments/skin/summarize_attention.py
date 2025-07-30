import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

demographic_groups = ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']

custom_colors = {
    'White': '#1f77b4',
    'Black': '#2ca02c',
    'Hispanic/Latino': '#ff7f0e',
    'Asian': '#d62728',
    'Other': '#9467bd',
    'Overall': '#8c564b'
}

custom_palette = {0: '#4dbbd5', 1: '#e64b35'}

demographic_group_order = [f'attn_{group}' for group in demographic_groups]

def bootstrap_ci(data, num_samples=1000, ci=95):
    """Generate bootstrap mean and confidence interval."""
    means = []
    n = len(data)
    for _ in range(num_samples):
        sample = np.random.choice(data, n, replace=True)
        means.append(np.mean(sample))
    lower_bound = np.percentile(means, (100 - ci) / 2)
    upper_bound = np.percentile(means, 100 - (100 - ci) / 2)
    return np.mean(means), lower_bound, upper_bound

def summarize_attention(base_dir, filter_threshold=15):
    # encoders = ['SP22M', 'uni', 'gigapath']
    encoders = ['SP22M']
    experiments = ['exp1','exp2','exp3']
    versions = ['', '_re_epi', '_keep_epi']

    total_iterations = len(encoders) * len(experiments)
    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for encoder in encoders:
            fig, axes = plt.subplots(2, len(experiments), figsize=(len(experiments) * 8, 10), sharey='row')
            all_samples = []
            summary_results = []
            for i, exp in enumerate(experiments):
                print(f"Processing encoder: {encoder}, experiment: {exp}")
                exp_path = Path(base_dir) / encoder / exp
                
                # Read marimo_meta.csv and use existing is_epithelium and attention scores
                marimo_meta_path = exp_path / 'marimo_metadata.csv'
                meta_df_sub = pd.read_csv(marimo_meta_path)
                meta_df_sub = meta_df_sub[['slide', 'x', 'y', 'is_epithelium'] + [f'attn_{group}' for group in demographic_groups]]
                
                # Filter out slides with less than 15 epithelium patches
                slides_of_interest = meta_df_sub['slide'].unique()
                epithelium_counts = meta_df_sub.groupby('slide')['is_epithelium'].sum()
                valid_slides = epithelium_counts[epithelium_counts >= filter_threshold].index
                removed_slides = set(slides_of_interest) - set(valid_slides)
                meta_df_sub = meta_df_sub[meta_df_sub['slide'].isin(valid_slides)]
                print(f"Number of slides removed due to not satisfying the threshold: {len(removed_slides)}")

                # Plot attention scores
                plot_attention_scores(meta_df_sub, axes[0, i], exp)

                # Sample slides for high/medium/low attention
                samples = sample_slides(meta_df_sub, exp)
                all_samples.append(samples)

                # Calculate and save summary statistics
                summary_results.extend(calculate_summary_statistics(meta_df_sub, exp))

                # Plot model performance comparison
                plot_model_performance_comparison(base_dir, encoder, exp, versions, axes[1, i])

                pbar.update(1)

            # Save combined sampled slides
            combined_samples = pd.concat(all_samples)
            combined_samples.to_csv(Path(base_dir) / encoder / 'combined_sampled_slides.csv', index=False)
            print(f"Combined sampled slides saved to {Path(base_dir) / encoder / 'combined_sampled_slides.csv'}")

            # Save combined plot
            plt.tight_layout()
            plt.savefig(Path(base_dir) / encoder / 'combined_attention_scores_and_performance.png', dpi=600, bbox_inches='tight')
            plt.close()
            print(f"Combined plot saved as {Path(base_dir) / encoder / 'combined_attention_scores_and_performance.png'}")

            # Save summary statistics to CSV
            summary_df = pd.DataFrame(summary_results)
            summary_df.to_csv(Path(base_dir) / encoder / 'attention_summary_statistics.csv', index=False)
            print(f"Summary statistics saved to {Path(base_dir) / encoder / 'attention_summary_statistics.csv'}")

def plot_attention_scores(meta_df_sub, ax, title):

    melted_df = meta_df_sub[['slide','is_epithelium','attn_White','attn_Black','attn_Hispanic/Latino','attn_Asian','attn_Other']].melt(
        id_vars=['slide', 'is_epithelium'], 
        value_vars=[f'attn_{race}' for race in demographic_groups],
        var_name='Race_Attn', 
        value_name='Attention Score'
    )
    
    melted_df = melted_df.groupby(['slide', 'is_epithelium', 'Race_Attn'])['Attention Score'].median().reset_index() # only take median attention score per slide

    sns.boxplot(x='Race_Attn', y='Attention Score', hue='is_epithelium', data=melted_df, order=demographic_group_order, ax=ax, palette=custom_palette)

    ax.set_xticks(range(len(demographic_groups)))
    ax.set_xticklabels([label.replace("attn_", "") for label in demographic_groups])
    
    max_y = melted_df['Attention Score'].max()
    # Add p-value annotations and brackets
    for i, race in enumerate(demographic_groups):
        non_epi_scores = melted_df.loc[(melted_df['Race_Attn'] == f'attn_{race}') & (melted_df['is_epithelium'] == 0), 'Attention Score']
        epi_scores = melted_df.loc[(melted_df['Race_Attn'] == f'attn_{race}') & (melted_df['is_epithelium'] == 1), 'Attention Score']
        p_val = ttest_ind(non_epi_scores, epi_scores, alternative='less', nan_policy='omit').pvalue
        if not np.isnan(p_val):
            if p_val < 0.05:
                # Determine significance level
                if p_val < 0.0001:
                    label = '***'
                elif p_val < 0.01:
                    label = '**'
                else:
                    label = '*'
                # Add star in red if epidermis is higher
                ax.text(i, max_y + 0.03, label, ha='center', fontsize=12, weight='bold', color='red')
                # Add bracket
                ax.plot([i - 0.2, i + 0.2], [max_y + 0.02, max_y + 0.02], color='black', linewidth=0.75)

    ax.set_ylim(0, 1)
    ax.set_xlabel('')
    if title == 'exp1':
        ax.set_ylabel('Median Attention Score Per Slide', fontsize=16, weight='bold')
    else:
        ax.set_ylabel('')

    # Add custom legend only for 'exp3'
    if title == 'exp3':
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ['Non-Epidermis', 'Epidermis'], loc='upper right', fontsize=18)
    else:
        ax.legend_.remove()

    ax.set_title(title, fontsize=18, weight='bold')

    # Set larger font size for x and y tick labels
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

def sample_slides(meta_df_sub, exp):
    samples = []
    for race in demographic_groups:
        for is_epithelium in [0, 1]:
            for interval, (low, high) in zip(['low', 'medium', 'high'], [(0, 25), (25, 75), (75, 100)]):
                subset = meta_df_sub[(meta_df_sub['is_epithelium'] == is_epithelium)]
                if not subset.empty:
                    low_threshold = np.percentile(subset[f'attn_{race}'], low)
                    high_threshold = np.percentile(subset[f'attn_{race}'], high)
                    sampled_slides = subset[(subset[f'attn_{race}'] >= low_threshold) & (subset[f'attn_{race}'] < high_threshold)].sample(n=3, replace=False, random_state=42)
                    sampled_slides['attn_interval'] = interval
                    sampled_slides['attn_race'] = race
                    sampled_slides['attn_is_epithelium'] = is_epithelium
                    sampled_slides['exp_version'] = exp
                    
                    samples.append(sampled_slides)

    sampled_df = pd.concat(samples)
    return sampled_df

def calculate_summary_statistics(meta_df_sub, exp):
    summary_results = []
    p_values = []
    for race in demographic_groups:
        non_epi_scores = meta_df_sub.loc[meta_df_sub['is_epithelium'] == 0, f'attn_{race}']
        epi_scores = meta_df_sub.loc[meta_df_sub['is_epithelium'] == 1, f'attn_{race}']
        t_stat, p_value = ttest_ind(non_epi_scores, epi_scores, alternative='less', nan_policy='omit')
        p_values.append(p_value)
        
        non_epi_mean, non_epi_ci_lower, non_epi_ci_upper = bootstrap_ci(non_epi_scores)
        epi_mean, epi_ci_lower, epi_ci_upper = bootstrap_ci(epi_scores)
        
        summary_results.append({
            'experiment': exp,
            'group': race,
            'non_epidermis_mean': non_epi_mean,
            'non_epidermis_ci_lower': non_epi_ci_lower,
            'non_epidermis_ci_upper': non_epi_ci_upper,
            'epidermis_mean': epi_mean,
            'epidermis_ci_lower': epi_ci_lower,
            'epidermis_ci_upper': epi_ci_upper,
            't_stat': t_stat,
            'p_value': p_value
        })
    
    # Adjust p-values using Bonferroni correction
    corrected_p_values = multipletests(p_values, method='bonferroni')[1]
    
    # Update summary_results with corrected p-values and significance labels
    for i, corrected_p_value in enumerate(corrected_p_values):
        summary_results[i]['corrected_p_value'] = corrected_p_value
        if corrected_p_value < 0.0001:
            summary_results[i]['significance'] = '***'
        elif corrected_p_value < 0.001:
            summary_results[i]['significance'] = '**'
        elif corrected_p_value < 0.05:
            summary_results[i]['significance'] = '*'
        else:
            summary_results[i]['significance'] = 'n.s.'
    
    return summary_results

def plot_model_performance_comparison(base_dir, encoder, exp, versions, ax):
    performance_data = []
    version_labels = {'': 'Original', '_re_epi': 'Remove Epidermis', '_keep_epi': 'Only with Epidermis'}
    for version in versions:
        exp_version = exp + version
        exp_path = Path(base_dir) / encoder / exp_version
        avg_probs_path = exp_path / 'average_probs.csv'
        if avg_probs_path.exists():
            probs_df = pd.read_csv(avg_probs_path)
            y_true = pd.get_dummies(probs_df['real'])[demographic_groups].values
            y_pred_proba = probs_df[demographic_groups].values
            for i, group in enumerate(demographic_groups):
                for _ in range(1000):  # Bootstrap 1000 samples
                    indices = np.random.choice(range(len(y_true)), len(y_true), replace=True)
                    auc_score = roc_auc_score(y_true[indices, i], y_pred_proba[indices, i], multi_class='ovr')
                    performance_data.append({
                        'Race': group,
                        'AUC': auc_score,
                        'Version': version_labels[version]
                    })

    performance_df = pd.DataFrame(performance_data)
    sns.boxplot(x='Race', y='AUC', hue='Version', data=performance_df, ax=ax, palette='Set2')
    
    # Statistical analysis for p-values
    for i, race in enumerate(demographic_groups):
        original = performance_df[(performance_df['Race'] == race) & (performance_df['Version'] == 'Original')]['AUC']
        re_epi = performance_df[(performance_df['Race'] == race) & (performance_df['Version'] == 'Remove Epidermis')]['AUC']
        keep_epi = performance_df[(performance_df['Race'] == race) & (performance_df['Version'] == 'Only with Epidermis')]['AUC']
        
        p_val_re_epi = ttest_ind(original, re_epi, alternative='greater').pvalue
        p_val_keep_epi = ttest_ind(original, keep_epi, alternative='greater').pvalue
        
        # # Adjust p-values using Bonferroni correction
        # p_vals = [p_val_re_epi, p_val_keep_epi]
        # corrected_p_vals = multipletests(p_vals, method='bonferroni')[1]
        # p_val_re_epi, p_val_keep_epi = corrected_p_vals
        
        max_y = performance_df['AUC'].max()
        min_y = performance_df['AUC'].min()
        
        if p_val_re_epi < 0.0001:
            label = '***'
        elif p_val_re_epi < 0.01:
            label = '**'
        elif p_val_re_epi < 0.05:
            label = '*'
        else:
            label = 'n.s.'
        
        if label != 'n.s.':
            ax.text(i - 0.125, max_y + 0.020, label, ha='center', fontsize=12, weight='bold', color='red')
        else:
            ax.text(i - 0.125, max_y + 0.020, label, ha='center', fontsize=12, weight='bold', color='black')
            
        ax.plot([i - 0.25, i], [max_y + 0.02, max_y + 0.02], color='black', linewidth=0.75) 
        ax.plot([i - 0.25, i - 0.25], [max_y + 0.02, max_y + 0.01], color='black', linewidth=0.75)  # Add left vertical line
        ax.plot([i, i], [max_y + 0.02, max_y + 0.01], color='black', linewidth=0.75)  # Add left vertical line
        
        if p_val_keep_epi < 0.0001:
            label = '***'
        elif p_val_keep_epi < 0.01:
            label = '**'
        elif p_val_keep_epi < 0.05:
            label = '*'
        else:
            label = 'n.s.'
            
        if label != 'n.s.':
            ax.text(i, max_y + 0.050, label, ha='center', fontsize=12, weight='bold', color='orange')
        else:
            ax.text(i, max_y + 0.050, label, ha='center', fontsize=12, weight='bold', color='black')
        ax.plot([i - 0.25, i + 0.25], [max_y + 0.05, max_y + 0.05], color='black', linewidth=0.75)
        ax.plot([i - 0.25, i - 0.25], [max_y + 0.05, max_y + 0.04], color='black', linewidth=0.75)  
        ax.plot([i + 0.25, i + 0.25], [max_y + 0.05, max_y + 0.04], color='black', linewidth=0.75)  
    
    ax.set_ylim(min_y-0.01, 1.0)
    
    # Set larger font size for x and y tick labels
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    
    if exp == 'exp2':
        ax.set_xlabel('Self Reported Race', fontsize=20, weight='bold')
    else:
        ax.set_xlabel('')
    ax.set_ylabel('AUC', fontsize=20, weight='bold')
    
    if exp == 'exp1':
        ax.legend(fontsize=18, loc='lower left')
    else:
        ax.legend_.remove()

if __name__ == "__main__":
    base_dir = 'CPath_SAIF/slide_experiments' 

    summarize_attention(base_dir)
