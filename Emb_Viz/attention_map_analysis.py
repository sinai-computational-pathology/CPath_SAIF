import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    from scipy.stats import entropy, kurtosis
    import yaml
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import pickle
    import matplotlib.pyplot as plt
    return Path, entropy, kurtosis, mo, np, pd, pickle, plt, yaml


@app.cell
def __(mo):
    mo.md("""# Load Data to Analysis""")
    return


@app.cell
def __(Path):
    base_path = Path('/slide_experiments/skin/')
    demographic_groups = ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']
    return base_path, demographic_groups


@app.cell
def __(mo):
    exps_options = ['exp1', 'exp2', 'exp3', 'exp4']

    mul_exps_dropdown = mo.ui.multiselect(
        options=exps_options,
        value=exps_options,
        label="Choose experiments to visualize"
    )

    mul_exps_dropdown
    return exps_options, mul_exps_dropdown


@app.cell
def __(mul_exps_dropdown):
    experiments = mul_exps_dropdown.value
    return (experiments,)


@app.cell
def __():
    ## get attention scores from all exps and calculate statistics metrics by slides
    return


@app.cell
def __(aggregate_metrics, experiments, process_experiment):
    aggregated_metrics_dict = {}
    attention_score_dicts = {}

    # Process each experiment in the list
    for exp in experiments:
        print(f'Processing {exp}...')
        attention_scores_dict, metrics_per_slide = process_experiment(exp)
        attention_score_dicts[exp] = attention_scores_dict
        aggregated_metrics_dict[exp] = aggregate_metrics(metrics_per_slide)
    return (
        aggregated_metrics_dict,
        attention_score_dicts,
        attention_scores_dict,
        exp,
        metrics_per_slide,
    )


@app.cell
def __(Path, demographic_groups, entropy, kurtosis, np, pickle):
    def load_attention_scores(exp_version):
        """Load attention scores dictionary."""
        base_path = Path(f'slide_experiments/skin/{exp_version}')
        attention_dict_file = base_path / 'attention_scores_dict.pkl'

        with open(attention_dict_file, 'rb') as f:
            attention_scores_dict = pickle.load(f)
        all_attention_scores = np.concatenate([scores.flatten() for scores in attention_scores_dict.values()])
        global_min = all_attention_scores.min()
        global_max = all_attention_scores.max()
        print(f"Loaded attention scores from {attention_dict_file}, global_min:{global_min},global_max:{global_max}")
        return attention_scores_dict, global_min, global_max
        
    def calculate_homogeneity_metrics(attention_scores):
        # Initialize dictionary to store homogeneity metrics for each race group
        metrics = {
            'entropy': {},
            'std_dev': {},
            'kurtosis': {}
        }

        # Calculate metrics for each race group independently
        for i, race_scores in enumerate(attention_scores):
            race_group = demographic_groups[i]

            # Ensure the scores sum to 1 by normalizing, to represent a probability distribution
            if race_scores.sum() > 0:
                race_probs = race_scores / race_scores.sum()
            else:
                # If all scores are zero, set a uniform distribution as fallback
                race_probs = np.ones_like(race_scores) / len(race_scores)

            # Clip to avoid log(0) issues in entropy calculation
            race_probs = np.clip(race_probs, 1e-10, None)

            # Calculate entropy, standard deviation, and kurtosis for the race group
            metrics['entropy'][race_group] = entropy(race_probs, base=2)
            metrics['std_dev'][race_group] = np.std(race_scores)
            metrics['kurtosis'][race_group] = kurtosis(race_scores)

        return metrics

    # Function to process all slides in an experiment and demographic group
    def process_experiment(exp_id):
        attention_scores_dict, _, _ = load_attention_scores(exp_id)
        metrics_per_slide = {}

        for slide_id, scores in attention_scores_dict.items():
            metrics = calculate_homogeneity_metrics(scores)
            metrics_per_slide[slide_id] = metrics

        return attention_scores_dict, metrics_per_slide

    def aggregate_metrics(metrics_per_slide):
        # Initialize dictionaries to hold aggregated metrics for each demographic group
        aggregated_metrics = {
            'entropy': {group: [] for group in demographic_groups},
            'std_dev': {group: [] for group in demographic_groups},
            'kurtosis': {group: [] for group in demographic_groups}
        }

        # Populate the aggregated metrics for each race group
        for slide_metrics in metrics_per_slide.values():
            for group in demographic_groups:
                aggregated_metrics['entropy'][group].append(slide_metrics['entropy'][group])
                aggregated_metrics['std_dev'][group].append(slide_metrics['std_dev'][group])
                aggregated_metrics['kurtosis'][group].append(slide_metrics['kurtosis'][group])

        return aggregated_metrics
    return (
        aggregate_metrics,
        calculate_homogeneity_metrics,
        load_attention_scores,
        process_experiment,
    )


@app.cell
def __(aggregated_metrics_dict, np, plt):
    def plot_distribution(aggregated_metrics_dict):
        # Demographic groups for labeling
        demographic_groups = ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']
        experiments = list(aggregated_metrics_dict.keys())  # List of experiment IDs

        # Define colors for each experiment's histogram (customize as needed)
        colors = ['skyblue', 'salmon', 'lightgreen', 'plum']

        # Define figure with 3 rows (one for each metric) and 5 columns (one for each race group)
        fig, axes = plt.subplots(3, 5, figsize=(20, 12), sharey=False)
        fig.suptitle('Homogeneity Metrics Distribution by Race Group Across Experiments', fontsize=16)

        # Metrics and their row indices
        metrics = ['entropy', 'std_dev', 'kurtosis']
        row_titles = ['Entropy', 'Std Dev', 'Kurtosis']

        # Calculate global x and y limits for each metric
        global_limits = {}
        for metric in metrics:
            all_values = []
            for exp_id in experiments:
                for group in demographic_groups:
                    all_values.extend(aggregated_metrics_dict[exp_id][metric][group])
            global_min_x, global_max_x = min(all_values), max(all_values)
            global_min_y, global_max_y = 0, 0  # Initialize y-axis limits

            # Estimate global y-axis limits based on density histograms
            for exp_id in experiments:
                for group in demographic_groups:
                    counts, bin_edges = np.histogram(
                        aggregated_metrics_dict[exp_id][metric][group],
                        bins=20,
                        density=True
                    )
                    global_max_y = max(global_max_y, counts.max())

            # Store limits for this metric
            global_limits[metric] = {
                'xlim': (global_min_x, global_max_x),
                'ylim': (global_min_y, global_max_y)
            }

        # Plot each metric for each demographic group
        for row_idx, metric in enumerate(metrics):
            for col_idx, group in enumerate(demographic_groups):
                # Plot histograms for all experiments
                for exp_id_idx, exp_id in enumerate(experiments):
                    axes[row_idx, col_idx].hist(
                        aggregated_metrics_dict[exp_id][metric][group],
                        bins=20,
                        color=colors[exp_id_idx],
                        alpha=0.5,
                        label=f'{exp_id}' if col_idx == 0 else "",  # Add legend only for the first column
                        edgecolor='black',
                        density=True
                    )

                # Set shared limits for the current metric
                axes[row_idx, col_idx].set_xlim(global_limits[metric]['xlim'])
                axes[row_idx, col_idx].set_ylim(global_limits[metric]['ylim'])

                # Add titles and labels
                axes[row_idx, col_idx].set_title(f'{group} - {row_titles[row_idx]}')
                axes[row_idx, col_idx].set_xlabel(row_titles[row_idx])
                axes[row_idx, col_idx].set_ylabel('Density' if col_idx == 0 else "")

        # Add legend for the first column in each row
        for row_idx, ax in enumerate(axes[:, 0]):
            ax.legend(title='Experiment')

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    # Visualize the aggregated metrics
    plot_distribution(aggregated_metrics_dict)
    return (plot_distribution,)


if __name__ == "__main__":
    app.run()
