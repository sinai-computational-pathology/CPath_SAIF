import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, recall_score, confusion_matrix
from pathlib import Path
import os
import pickle

# Define parameters
confidence_level = 0.95
class_prob_columns = ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']

custom_colors = {
    'White': '#1f77b4',
    'Black': '#2ca02c',
    'Hispanic/Latino': '#ff7f0e',
    'Asian': '#d62728',
    'Other': '#9467bd',
    'Overall': '#8c564b'
}

def calculate_stats_multiclass(target_class, predicted_class, average="macro"):
    accuracy = accuracy_score(target_class, predicted_class)
    recall = recall_score(target_class, predicted_class, average=average)
    cm = confusion_matrix(target_class, predicted_class)
    
    specificity_per_class = []
    num_classes = cm.shape[0]
    for i in range(num_classes):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)

    return accuracy, recall, np.mean(specificity_per_class)


# Function to find representative examples based on race, including detailed analysis
def find_representative_examples_by_race(df, num_samples=5):
    class_columns = ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']
    results = []
    
    # pooling_methods = ['average', 'max', 'median']
    
    # # 1. Tile-level analysis for most accurate predictions for each race
    # for race in class_columns:
    #     race_group = df[df['real'] == race]

    #     # Find most accurate examples for each race
    #     accuracy_by_slide = race_group.groupby('slide')['correct'].mean().reset_index()
    #     top_accurate_slides = accuracy_by_slide.nlargest(num_samples, 'correct')['slide']

    #     for slide in top_accurate_slides:
    #         slide_df = race_group[race_group['slide'] == slide]
    #         accuracy, sensitivity, specificity = calculate_stats_multiclass(
    #             target_class=slide_df['real'],
    #             predicted_class=slide_df['pred']
    #         )
    #         # Use slide-level prediction as same as the target for this case
    #         slide_level_pred = slide_df['real'].iloc[0]

    #         results.append({
    #             'slide': slide,
    #             'race': slide_df['real'].iloc[0],
    #             'organ': slide_df['organ'].iloc[0],
    #             'comment': f'Most accurate predictions for race: {race} ',
    #             'accuracy': accuracy,
    #             'sensitivity': sensitivity,
    #             'specificity': specificity,
    #             'pred': slide_level_pred,  # Pred is same as target here
    #             'target': slide_df['real'].iloc[0],  # Target from the first row
    #             'pooling_method': 'exact',  # Exact match since it's the most accurate prediction
    #         })

    # 2. Correct Predictions at Slide-Level (using aggregate_predictions)
    # for method in pooling_methods:
    slide_level_df = df
    
    for race in class_columns:
        correct_condition = (slide_level_df['pred'] == race) & (slide_level_df['real'] == race)
        correct_slides = slide_level_df[correct_condition].drop_duplicates(subset='slide').sample(
            min(len(slide_level_df[correct_condition].drop_duplicates(subset='slide')), num_samples), random_state=42)

        if len(correct_slides) < num_samples:
            print(f"Warning: Found only {len(correct_slides)} correct predictions for race: {race}")

        for _, slide_sample in correct_slides.iterrows():
            slide_df = df[df['slide'] == slide_sample['slide']]
            accuracy, sensitivity, specificity = calculate_stats_multiclass(
                target_class=slide_df['real'],
                predicted_class=slide_df['pred']
            )
            results.append({
                'slide': slide_sample['slide'],
                'race': slide_sample['real'],
                'organ': slide_sample['organ'],
                'comment': f'Correct prediction for race: {race} at slide-level',
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'pred': slide_sample['pred'],  # Slide-level prediction
                'target': slide_sample['real'],  # Target
            })

    # 3. False Positive (predicted race != real race)
    for real_race in class_columns:
        for pred_race in class_columns:
            if real_race == pred_race:
                continue
            
            slide_level_df = df
            false_positive_condition = (slide_level_df['pred'] == pred_race) & (slide_level_df['real'] == real_race)
            false_positive_slides = slide_level_df[false_positive_condition].drop_duplicates(subset='slide').sample(
                min(len(slide_level_df[false_positive_condition].drop_duplicates(subset='slide')), num_samples), random_state=42)
            
            if len(false_positive_slides) < num_samples:
                print(f"Warning: Found only {len(false_positive_slides)} false positives for real race: {real_race}, predicted as {pred_race}")

            for _, slide_sample in false_positive_slides.iterrows():
                slide_df = df[df['slide'] == slide_sample['slide']]
                accuracy, sensitivity, specificity = calculate_stats_multiclass(
                    target_class=slide_df['real'],
                    predicted_class=slide_df['pred']
                )
                results.append({
                    'slide': slide_sample['slide'],
                    'race': slide_sample['real'],
                    'organ': slide_sample['organ'],
                    'comment': f'Self-reported race was {real_race}, predicted as {pred_race} at slide-level',
                    'accuracy': accuracy,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'pred': slide_sample['pred'],  # Slide-level prediction
                    'target': slide_sample['real'],  # Target
                })

    # Return final results
    final_results = pd.DataFrame(results)
    return final_results

# Function to find representative examples based on organ, focusing on general false positives, false negatives, and correct predictions
def find_representative_examples_by_organ(df, num_samples=3):
    results = []

    # Tile-level analysis for organs
    for organ in df['organ'].unique():
        # Filter data for the specific race (slide-level analysis)
        organ_group = df[df['organ'] == organ]
        df_results = find_representative_examples_by_race(organ_group, num_samples=num_samples)
        results.append(df_results)

    final_results = pd.concat(results, axis=0).reset_index(drop=True)
    return final_results
    

# Function to calculate and plot AUC or Accuracy by organ
def plot_metrics_by_organ(probs_df, output_folder, task_name):
    print(f"Calculating and plotting AUC/Accuracy for each organ...")
    
    class_names = class_prob_columns  # Assuming this is defined elsewhere
    metrics_by_organ = {}

    # Get the value counts for each organ and print them
    # organ_value_counts = probs_df.groupby('organ')['real'].value_counts()
    # print("Value counts of each organ:\n", organ_value_counts)

    plot_race_distribution_by_organ(probs_df, output_folder, task_name)
    
    # Group by organ and calculate AUC or Accuracy for each organ
    organs = probs_df['organ'].unique()
    palette = sns.color_palette('Set2', n_colors=len(organs))

    print("organ value counts before plotting in each level:", probs_df['organ'].value_counts())

    for organ in organs:
        organ_df = probs_df[probs_df['organ'] == organ]
        true_labels = organ_df['real'].values
        unique_labels = np.unique(true_labels)
        
        if len(unique_labels) == 1:
            # Only one class present, calculate Accuracy
            predicted_labels = organ_df[class_names].idxmax(axis=1).values
            accuracy = accuracy_score(true_labels, predicted_labels)
            metrics_by_organ[organ] = {'metric': 'Accuracy', 'value': accuracy}
            print(f"Organ '{organ}' has only one class '{unique_labels[0]}'. Calculated Accuracy: {accuracy:.4f}")
        else:
            # Multiple classes present, calculate AUC
            auc_scores_per_class = []
            for class_name in class_names:
                probabilities = organ_df[class_name].values
                binary_labels = (true_labels == class_name).astype(int)
                if len(np.unique(binary_labels)) < 2:
                    # Skip if this class is not present in true labels
                    continue
                auc = roc_auc_score(binary_labels, probabilities)
                auc_scores_per_class.append(auc)
            
            if auc_scores_per_class:
                # Store the mean AUC for the organ
                mean_auc_organ = np.mean(auc_scores_per_class)
                metrics_by_organ[organ] = {'metric': 'AUC', 'value': mean_auc_organ}
                print(f"Organ '{organ}' AUC: {mean_auc_organ:.4f}")
            else:
                # No AUC could be calculated
                print(f"Organ '{organ}' has insufficient class diversity for AUC calculation.")
                metrics_by_organ[organ] = {'metric': 'AUC', 'value': np.nan}
    
    # Prepare data for plotting
    organs = list(metrics_by_organ.keys())
    metric_values = [metrics_by_organ[organ]['value'] for organ in organs]
    metrics = [metrics_by_organ[organ]['metric'] for organ in organs]
    
    # Plot the metrics for each organ
    plt.figure(figsize=(12, 7))
    bars = plt.bar(organs, metric_values, color=palette, alpha=0.7)
    plt.ylabel('Metric Value')
    plt.title(f'AUC by Organ for {task_name}')
    plt.xticks(rotation=45)
    
    # # Annotate bars with metric type
    # for bar, metric in zip(bars, metrics):
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2, height, f'{metric}', ha='center', va='bottom')
    
    # Save the plot
    plot_file = output_folder / f'{task_name}_metrics_by_organ.png'
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    print(f"AUC/Accuracy by organ plot saved as {plot_file}")

# Function to plot race distribution pie charts for each organ
def plot_race_distribution_by_organ(probs_df, output_folder, task_name):
    print("Plotting race distribution for each organ...")
    
    # Get the list of organs
    organs = probs_df['organ'].unique()
    num_organs = len(organs)
    
    # Determine grid size for subplots
    cols = 4 
    rows = (num_organs + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()
    
    for i, organ in enumerate(organs):
        organ_df = probs_df[probs_df['organ'] == organ]
        race_counts = organ_df['real'].value_counts()
        # Prepare data for pie chart
        labels = race_counts.index
        sizes = race_counts.values
        colors = [custom_colors[race] for race in labels]
        
        # Plot pie chart
        ax = axes[i]
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops=dict(color="black")
        )
        ax.axis('equal')
        ax.set_title(f'Organ: {organ}')
    
    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    # Save the figure
    plot_file = output_folder / f'{task_name}_race_distribution_by_organ.png'
    plt.savefig(plot_file)
    plt.close()
    print(f"Race distribution by organ plot saved as {plot_file}")

# # Function to plot heatmap for the representative examples
# def plot_results_heatmap(results_df, df, level):
#     heatmap_data = []
#     class_labels = ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']

#     for _, row in results_df.iterrows():
#         entity = row[level]  
#         comment = row['comment'] 
#         accuracy = row['accuracy']

#         slide_df = df[df[level] == entity]
#         avg_probs = slide_df[class_labels].mean().values
#         heatmap_data.append(avg_probs)

#     heatmap_df = pd.DataFrame(heatmap_data, columns=class_labels)

#     # Add the comments and accuracy as row labels
#     heatmap_df.index = results_df.apply(lambda x: f"{x['comment']} (Acc: {x['accuracy']:.2f})", axis=1)

#     # Plot the heatmap
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(heatmap_df, annot=True, cmap='Blues', fmt='.2f', linewidths=.5)

#     # Title and labels
#     plt.title(f"Predicted Class Probabilities for {level.capitalize()} Examples")
#     plt.xlabel("Predicted Class")
#     plt.ylabel(f"{level.capitalize()} (Comment and Accuracy)")
#     plt.show()

# Function to aggregate tile-level predictions to slide-level or patient-level
def aggregate_predictions(df, level='slide', pooling='average'):
    class_prob_columns = ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']
    
    # Choose the column to group by based on the specified level
    group_by_col = 'slide' if level == 'slide' else 'MRN'

    # Group by the chosen level and apply pooling (mean or max) for class probabilities
    if pooling == 'average':
        aggregated_df = df.groupby(group_by_col)[class_prob_columns].mean().reset_index()
    elif pooling == 'max':
        aggregated_df = df.groupby(group_by_col)[class_prob_columns].max().reset_index()
    else:
        raise ValueError("Pooling must be 'average' or 'max'.")

    # Assign the 'real' target by taking the first value of 'real' for each group
    aggregated_df['real'] = df.groupby(group_by_col)['real'].first().values
    aggregated_df['target'] = df.groupby(group_by_col)['target'].first().values
    aggregated_df['organ'] = df.groupby(group_by_col)['organ'].first().values
    aggregated_df['pred'] = aggregated_df[class_prob_columns].idxmax(axis=1)
    
    return aggregated_df

def map_slide_to_patient_and_organ(slide_level_df, mapping_file):
    """
    Maps each slide in slide-level DataFrame to the corresponding patient (MRN) and organ
    based on a mapping file. Reports any inconsistencies where a slide is associated with more than 
    one MRN or more than one organ.
    
    Parameters:
    - slide_level_df: DataFrame containing slide-level data
    - mapping_file: Path to the CSV file with MRN-slide-organ mapping
    
    Returns:
    - A new DataFrame with slides mapped to MRNs and organs
    """
    # Load the slide-to-MRN and organ mapping file
    mapping_df = pd.read_csv(mapping_file)
    
    # print("before merge_df organ:",mapping_df['organ'].value_counts())
    # Merge slide-level DataFrame with the MRN and organ mapping
    merged_df = pd.merge(slide_level_df, mapping_df[['MRN', 'slide', 'organ']], on='slide', how='left')
    # print("after merge_df organ:", merged_df['organ'].value_counts())

    # Check for any inconsistencies where a slide has more than one MRN
    duplicate_mrn_slides = mapping_df.groupby('slide')['MRN'].nunique()
    problematic_mrn_slides = duplicate_mrn_slides[duplicate_mrn_slides > 1]

    if not problematic_mrn_slides.empty:
        print("Warning: The following slides are associated with more than one MRN:")
        print(problematic_mrn_slides)
    
    # Check for any inconsistencies where a slide has more than one organ
    duplicate_organ_slides = mapping_df.groupby('slide')['organ'].nunique()
    problematic_organ_slides = duplicate_organ_slides[duplicate_organ_slides > 1]

    if not problematic_organ_slides.empty:
        print("Warning: The following slides are associated with more than one organ:")
        print(problematic_organ_slides)

    return merged_df


# Function for bootstrapping and AUC calculation for each class (one-vs-rest)
def bootstrap_ovr_auc_ci(probs_df, output_file, n_iterations=1000, confidence_level=0.95, level='tile', pooling='average'):
    
    class_names = class_prob_columns
    auc_scores_per_class = []
    mean_auc_per_class = []
    
    true_labels = probs_df['real'].values  # Ground truth target class labels
    n = len(probs_df)
    
    # Iterate through each class (one-vs-rest)
    for class_name in class_names:
        auc_scores = []
        probabilities = probs_df[class_name].values
        
        for _ in range(n_iterations):
            # Bootstrap sampling
            indices = np.random.choice(np.arange(n), size=int(n * args.ratio_samples), replace=True)
            if len(np.unique(true_labels[indices])) > 1:
                # Convert true_labels to binary for one-vs-rest classification
                binary_labels = (true_labels[indices] == class_name).astype(int)
                auc = roc_auc_score(binary_labels, probabilities[indices])
                auc_scores.append(auc)
        
        auc_scores_per_class.append(auc_scores)
        mean_auc_per_class.append(np.mean(auc_scores))
    
    # Calculate overall AUC (average AUC of all classes per iteration)
    auc_scores_overall = [np.mean([auc_scores_per_class[i][j] for i in range(len(class_names))]) for j in range(n_iterations)]
    mean_auc_overall = np.mean(auc_scores_overall)
    lower_bound = np.percentile(auc_scores_overall, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(auc_scores_overall, (1 + confidence_level) / 2 * 100)
    ci_ovr = (lower_bound, upper_bound)
    
    # Save the results to CSV
    auc_df = pd.DataFrame(auc_scores_per_class).T
    auc_df.columns = class_names
    auc_df['Overall'] = auc_scores_overall
    auc_df.to_csv(output_file, index=False)
    
    return auc_scores_per_class, mean_auc_per_class, auc_scores_overall, ci_ovr

# Function to plot AUC distribution (bar plot with error bars and boxplot)
def plot_auc_distribution(auc_scores_per_class, mean_auc_per_class, auc_scores_overall, mean_auc_overall, output_folder, task_name):
    class_names = class_prob_columns
    class_names_with_overall = class_names + ['Overall']
    mean_auc_all = list(mean_auc_per_class) + [mean_auc_overall]
    
    # Combine all the AUC scores for the boxplot
    auc_scores_combined = auc_scores_per_class + [auc_scores_overall]
    
    # Calculate 95% confidence intervals
    ci_lower = []
    ci_upper = []
    for auc_scores in auc_scores_combined:
        lower_bound = np.percentile(auc_scores, 2.5)
        upper_bound = np.percentile(auc_scores, 97.5)
        ci_lower.append(mean_auc_all[len(ci_lower)] - lower_bound)
        ci_upper.append(upper_bound - mean_auc_all[len(ci_upper)])
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = [custom_colors[class_name] for class_name in class_names_with_overall]

    # Bar plot with error bars
    bars = axes[0].bar(class_names_with_overall, mean_auc_all, color=colors, alpha=0.5, label='Mean AUC')
    axes[0].errorbar(class_names_with_overall, mean_auc_all, yerr=[ci_lower, ci_upper], fmt='none', color='black', capsize=5)
    axes[0].set_ylabel('AUC Score')
    axes[0].set_title(f'AUC Distribution for {task_name}')
    axes[0].set_xticklabels(class_names_with_overall, rotation=45)
    axes[0].set_ylim(0.5, 1.0)
    
    # Boxplot
    auc_scores_flat = [score for auc_scores in auc_scores_combined for score in auc_scores]
    class_labels = [label for label, auc_scores in zip(class_names_with_overall, auc_scores_combined) for _ in auc_scores]

    palette = [custom_colors[class_name] for class_name in class_names_with_overall]
    sns.boxplot(x=class_labels, y=auc_scores_flat, ax=axes[1], palette=palette)
    axes[1].set_title(f'AUC Distribution (Boxplot) for {task_name}')
    axes[1].set_xticklabels(class_names_with_overall, rotation=45)
    
    # Save the plot
    plot_file = output_folder / f'{task_name}_auc_distribution.png'
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    print(f"Plot saved to {plot_file}")

# Function to plot confusion matrix heatmap
def plot_confusion_matrix(probs_df, dataset_name, output_folder, level_name):
    # Create confusion matrix, normalized by row (as percentage)
    cm = pd.crosstab(probs_df['real'], probs_df['pred'], normalize='index')  # Convert to percentage
    
    plt.figure(figsize=(8, 8))
    
    # Create heatmap with percentage values and better labels
    sns.heatmap(cm, annot=True, fmt='.2f', cbar_kws={'label': 'Percentage (%)'}, 
            annot_kws={"fontsize": 12, "fontweight": "bold"})
    
    # Set the axis labels and title
    plt.xlabel('Predicted Race', fontsize=12, fontweight='bold')
    plt.ylabel('Self-Reported Race', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix Heatmap for {level_name} (Percentage)', fontsize=14, fontweight='bold')
    
    # Save the plot
    plt.savefig(output_folder / f'confusion_matrix_{dataset_name}_{level_name}.png')
    plt.close()
    print(f"Confusion Matrix saved for {dataset_name} - {level_name}")


# Function to plot ROC curves for each class
def plot_roc_curve(probs_df, dataset_name, output_folder, level_name):
    fig, ax = plt.subplots(figsize=(10, 8))
    for group in class_prob_columns:
        y_true = (probs_df['real'] == group).astype(int)
        y_score = probs_df[group]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        color = custom_colors.get(group, '#000000')
        ax.plot(fpr, tpr, label=f'{group} (AUC = {roc_auc:.2f}) - {dataset_name}', color=color)

    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves for {dataset_name} - {level_name}')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_folder / f'roc_curve_{dataset_name}_{level_name}.png')
    plt.close()

    print(f"ROC curve saved for {dataset_name} - {level_name}")

# Main function that runs bootstrapping and plotting
def run_bootstrap_auc():
    input_file = Path(args.input)
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    mapping_file = '/sc/arion/projects/comppath_SAIF/data/cohort_08_15_2024/MTL110623_master.csv'

    # Load probabilities and true labels (tile-level)
    probs_df = pd.read_csv(input_file)
    probs_df['real'] = probs_df.target.map({0: 'White', 1: 'Black', 2: 'Hispanic/Latino', 3: 'Asian', 4: 'Other'})
    probs_df['pred'] = probs_df[['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']].idxmax(1)
    probs_df['correct'] = probs_df['pred'] == probs_df['real'] 
    probs_df = map_slide_to_patient_and_organ(probs_df, mapping_file)

    # Perform task-specific AUC computation and plotting
    # if args.task == 'tile_level':
    #     run_bootstrap_for_level(probs_df, output_folder, 'tile_level', 'tile_level')
    #     race_results_df = find_representative_examples_by_race(probs_df, num_samples=5)
    #     organ_results_df = find_representative_examples_by_organ(probs_df, num_samples=5)
    #     merged_results_df = pd.concat([race_results_df, organ_results_df], axis=0).reset_index(drop=True)
    #     merged_results_df.to_csv(output_folder / f'{args.task}_examples.csv')
        # plot_results_heatmap(tile_results_df, probs_df, level='tile')

    if args.task == 'slide_level':
        run_bootstrap_for_level(probs_df, output_folder, 'slide_level', 'slide_level')
        race_results_df = find_representative_examples_by_race(probs_df, num_samples=5)
        # organ_results_df = find_representative_examples_by_organ(probs_df, num_samples=5)
        # merged_results_df = pd.concat([race_results_df, organ_results_df], axis=0).reset_index(drop=True)
        race_results_df.to_csv(output_folder / f'{args.task}_examples.csv')
        # organ_results_df.to_csv(output_folder / f'{args.task}_examples_by_organs.csv')
        # plot_results_heatmap(tile_results_df, probs_df, level='tile')

    elif args.task == 'slide_level_avg_pooling':
        # Aggregate to slide-level with average pooling
        aggregated_df = aggregate_predictions(probs_df, level='slide', pooling='average')
        run_bootstrap_for_level(aggregated_df, output_folder, 'slide_level', 'slide_level_avg_pooling', pooling='average')
        # slide_results_df = find_representative_examples(aggregated_df, level='slide')
        # slide_results_df.to_csv(output_folder / f'{args.task}_examples.csv')
        # plot_results_heatmap(slide_results_df, aggregated_df, level='slide')

    elif args.task == 'slide_level_max_pooling':
        # Aggregate to slide-level with max pooling
        aggregated_df = aggregate_predictions(probs_df, level='slide', pooling='max')
        run_bootstrap_for_level(aggregated_df, output_folder, 'slide_level', args.task, pooling='max')
        # slide_results_df = find_representative_examples(aggregated_df, level='slide')
        # slide_results_df.to_csv(output_folder / f'{args.task}_examples.csv')
        # plot_results_heatmap(slide_results_df, aggregated_df, level='slide')

    elif args.task == 'patient_level_avg_pooling':
        # Aggregate to patient-level with average pooling
        aggregated_df = aggregate_predictions(probs_df, level='patient', pooling='average')
        run_bootstrap_for_level(aggregated_df, output_folder, 'patient_level', args.task, pooling='average')
        # patient_results_df = find_representative_examples(aggregated_df, level='patient')
        # patient_results_df.to_csv(output_folder / f'{args.task}_examples.csv')
        # plot_results_heatmap(patient_results_df, aggregated_df, level='patient')

    elif args.task == 'patient_level_max_pooling':
        # Aggregate to patient-level with max pooling
        aggregated_df = aggregate_predictions(probs_df, level='patient', pooling='max')
        run_bootstrap_for_level(aggregated_df, output_folder, 'patient_level', args.task, pooling='max')
        # patient_results_df = find_representative_examples(aggregated_df, level='patient')
        # patient_results_df.to_csv(output_folder / f'{args.task}_examples.csv')
        # plot_results_heatmap(patient_results_df, aggregated_df, level='patient')

    else:
        print("Invalid task specified. Choose from 'tile_level', 'slide_level_avg_pooling', 'slide_level_max_pooling', 'patient_level_avg_pooling', 'patient_level_max_pooling'.")

def run_bootstrap_for_level(probs_df, output_folder, level, task_name, pooling='average'):
    results_file = output_folder / f'{task_name}_bootstrap_results.pkl'
    output_file = output_folder / f'{task_name}_auc_results.csv'

    # Perform confusion matrix and ROC curve plotting
    plot_confusion_matrix(probs_df, '21M', output_folder, task_name)
    plot_roc_curve(probs_df, '21M', output_folder, task_name)
    if "notskin" in str(output_folder):
        plot_metrics_by_organ(probs_df, output_folder, task_name)
    
    # Check if bootstrapped results already exist to avoid recomputation
    if results_file.exists():
        print(f"Loading precomputed {task_name} bootstrap results...")
        with open(results_file, 'rb') as f:
            auc_scores_per_class, mean_auc_per_class, auc_scores_overall, ci_ovr = pickle.load(f)
    else:
        print(f"Calculating {level} AUC with {pooling} pooling...")
        auc_scores_per_class, mean_auc_per_class, auc_scores_overall, ci_ovr = bootstrap_ovr_auc_ci(
            probs_df, output_file, n_iterations=args.n_iterations, confidence_level=confidence_level, level=level, pooling=pooling
        )
        with open(results_file, 'wb') as f:
            pickle.dump([auc_scores_per_class, mean_auc_per_class, auc_scores_overall, ci_ovr], f)
    
    print(f"{task_name} OVR AUC: {np.mean(auc_scores_overall):.3f}, 95% CI: {ci_ovr}")
    plot_auc_distribution(auc_scores_per_class, mean_auc_per_class, auc_scores_overall, np.mean(auc_scores_overall), output_folder, task_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bootstrap AUC with 95% CI calculation and plotting.')
    parser.add_argument('--task', type=str, required=True, help="Specify the task (e.g., 'tile_level', 'slide_level_avg_pooling', 'patient_level_max_pooling').")
    parser.add_argument('--input', type=str, required=True, help='Path to the input probability file (CSV).')
    parser.add_argument('--output', type=str, required=True, help='Directory to save the results.')
    parser.add_argument('--n_iterations', type=int, default=1000, help='Number of bootstrapping iterations.')
    parser.add_argument('--ratio_samples', type=float, default=1.0, help='Ratio of samples to use in bootstrapping.')

    global args
    args = parser.parse_args()

    run_bootstrap_auc()