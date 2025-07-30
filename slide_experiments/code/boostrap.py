import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, recall_score, confusion_matrix, f1_score, precision_score, matthews_corrcoef, cohen_kappa_score
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
    precision = precision_score(target_class, predicted_class, average=average)
    f1 = f1_score(target_class, predicted_class, average=average)
    cm = confusion_matrix(target_class, predicted_class)
    
    specificity_per_class = []
    num_classes = cm.shape[0]
    for i in range(num_classes):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)

    return accuracy, recall, precision, f1, np.mean(specificity_per_class)

def find_representative_examples_by_race(df, num_samples=5):
    class_columns = ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']
    results = []
    
    for race in class_columns:
        correct_condition = (df['pred'] == race) & (df['real'] == race)
        correct_slides_df = df[correct_condition].drop_duplicates(subset='slide')
        correct_slides = correct_slides_df.sample(
            min(len(correct_slides_df), num_samples), random_state=42)
        if len(correct_slides) < num_samples:
            print(f"Warning: Found only {len(correct_slides)} correct predictions for race: {race}")

        for _, slide_sample in correct_slides.iterrows():
            results.append({
                'slide': slide_sample['slide'],
                'race': slide_sample['real'],
                'organ': slide_sample['organ'],
                'comment': f'Correct prediction for race: {race} at slide-level',
                'pred': slide_sample['pred'],  # Slide-level prediction
                'target': slide_sample['real'],  # Target
            })

    for real_race in class_columns:
        for pred_race in class_columns:
            if real_race == pred_race:
                continue
            
            false_positive_condition = (df['pred'] == pred_race) & (df['real'] == real_race)
            false_positive_slides_df = df[false_positive_condition].drop_duplicates(subset='slide')
            false_positive_slides = false_positive_slides_df.sample(
                min(len(false_positive_slides_df), num_samples), random_state=42)
            if len(false_positive_slides) < num_samples:
                print(f"Warning: Found only {len(false_positive_slides)} false positives for real race: {real_race}, predicted as {pred_race}")

            for _, slide_sample in false_positive_slides.iterrows():
                results.append({
                    'slide': slide_sample['slide'],
                    'race': slide_sample['real'],
                    'organ': slide_sample['organ'],
                    'comment': f'Self-reported race was {real_race}, predicted as {pred_race} at slide-level',
                    'pred': slide_sample['pred'],  # Slide-level prediction
                    'target': slide_sample['real'],  # Target
                })

    final_results = pd.DataFrame(results)
    return final_results

def map_slide_to_patient_and_organ(slide_level_df, mapping_file):
    mapping_df = pd.read_csv(mapping_file)
    merged_df = pd.merge(slide_level_df, mapping_df[['MRN', 'slide', 'organ']], on='slide', how='left')

    duplicate_mrn_slides = mapping_df.groupby('slide')['MRN'].nunique()
    problematic_mrn_slides = duplicate_mrn_slides[duplicate_mrn_slides > 1]

    if not problematic_mrn_slides.empty:
        print("Warning: The following slides are associated with more than one MRN:")
        print(problematic_mrn_slides)
    
    duplicate_organ_slides = mapping_df.groupby('slide')['organ'].nunique()
    problematic_organ_slides = duplicate_organ_slides[duplicate_organ_slides > 1]

    if not problematic_organ_slides.empty:
        print("Warning: The following slides are associated with more than one organ:")
        print(problematic_organ_slides)

    return merged_df

def bootstrap_ovr_auc_ci(probs_df, output_file, n_iterations=1000, confidence_level=0.95, level='tile', pooling='average'):
    class_names = class_prob_columns
    auc_scores_per_class = []
    mean_auc_per_class = []
    
    true_labels = probs_df['real'].values
    n = len(probs_df)
    
    for class_name in class_names:
        auc_scores = []
        probabilities = probs_df[class_name].values
        
        for _ in range(n_iterations):
            indices = np.random.choice(np.arange(n), size=int(n * args.ratio_samples), replace=True)
            if len(np.unique(true_labels[indices])) > 1:
                binary_labels = (true_labels[indices] == class_name).astype(int)
                auc = roc_auc_score(binary_labels, probabilities[indices])
                auc_scores.append(auc)
        
        auc_scores_per_class.append(auc_scores)
        mean_auc_per_class.append(np.mean(auc_scores))
    
    auc_scores_overall = [np.mean([auc_scores_per_class[i][j] for i in range(len(class_names))]) for j in range(n_iterations)]
    mean_auc_overall = np.mean(auc_scores_overall)
    lower_bound = np.percentile(auc_scores_overall, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(auc_scores_overall, (1 + confidence_level) / 2 * 100)
    ci_ovr = (lower_bound, upper_bound)
    
    auc_df = pd.DataFrame(auc_scores_per_class).T
    auc_df.columns = class_names
    auc_df['Overall'] = auc_scores_overall
    auc_df.to_csv(output_file, index=False)
    
    return auc_scores_per_class, mean_auc_per_class, auc_scores_overall, ci_ovr

def bootstrap_metrics(probs_df, n_iterations=1000, confidence_level=0.95):
    accuracy_scores = []
    f1_scores = []
    
    true_labels = probs_df['real'].values
    predicted_labels = probs_df['pred'].values
    n = len(probs_df)
    
    for _ in range(n_iterations):
        indices = np.random.choice(np.arange(n), size=n, replace=True)
        accuracy = accuracy_score(true_labels[indices], predicted_labels[indices])
        f1 = f1_score(true_labels[indices], predicted_labels[indices], average='macro')
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
    
    mean_accuracy = np.mean(accuracy_scores)
    mean_f1 = np.mean(f1_scores)
    
    accuracy_ci = (np.percentile(accuracy_scores, (1 - confidence_level) / 2 * 100),
                   np.percentile(accuracy_scores, (1 + confidence_level) / 2 * 100))
    
    f1_ci = (np.percentile(f1_scores, (1 - confidence_level) / 2 * 100),
             np.percentile(f1_scores, (1 + confidence_level) / 2 * 100))
    
    return mean_accuracy, accuracy_ci, mean_f1, f1_ci

def plot_auc_distribution(auc_scores_per_class, mean_auc_per_class, auc_scores_overall, mean_auc_overall, output_folder, task_name):
    class_names = class_prob_columns
    class_names_with_overall = class_names + ['Overall']
    mean_auc_all = list(mean_auc_per_class) + [mean_auc_overall]
    
    auc_scores_combined = auc_scores_per_class + [auc_scores_overall]
    
    ci_lower = []
    ci_upper = []
    for auc_scores in auc_scores_combined:
        lower_bound = np.percentile(auc_scores, 2.5)
        upper_bound = np.percentile(auc_scores, 97.5)
        ci_lower.append(mean_auc_all[len(ci_lower)] - lower_bound)
        ci_upper.append(upper_bound - mean_auc_all[len(ci_upper)])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = [custom_colors[class_name] for class_name in class_names_with_overall]

    bars = axes[0].bar(class_names_with_overall, mean_auc_all, color=colors, alpha=0.5, label='Mean AUC')
    axes[0].errorbar(class_names_with_overall, mean_auc_all, yerr=[ci_lower, ci_upper], fmt='none', color='black', capsize=5)
    axes[0].set_ylabel('AUC Score')
    axes[0].set_title(f'AUC Distribution for {task_name}')
    axes[0].set_xticklabels(class_names_with_overall, rotation=45)
    axes[0].set_ylim(0.5, 1.0)
    
    auc_scores_flat = [score for auc_scores in auc_scores_combined for score in auc_scores]
    class_labels = [label for label, auc_scores in zip(class_names_with_overall, auc_scores_combined) for _ in auc_scores]

    palette = [custom_colors[class_name] for class_name in class_names_with_overall]
    sns.boxplot(x=class_labels, y=auc_scores_flat, ax=axes[1], palette=palette)
    axes[1].set_title(f'AUC Distribution (Boxplot) for {task_name}')
    axes[1].set_xticklabels(class_names_with_overall, rotation=45)
    
    plot_file = output_folder / f'{task_name}_auc_distribution.png'
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    print(f"Plot saved to {plot_file}")

def plot_confusion_matrix(probs_df, dataset_name, output_folder, level_name):
    cm = pd.crosstab(probs_df['real'], probs_df['pred'], normalize='index')
    
    plt.figure(figsize=(8, 8))
    
    sns.heatmap(cm, annot=True, fmt='.2f', cbar_kws={'label': 'Percentage (%)'}, 
            annot_kws={"fontsize": 12, "fontweight": "bold"})
    
    plt.xlabel('Predicted Race', fontsize=12, fontweight='bold')
    plt.ylabel('Self-Reported Race', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix Heatmap for {level_name} (Percentage)', fontsize=14, fontweight='bold')
    
    plt.savefig(output_folder / f'confusion_matrix_{dataset_name}_{level_name}.png')
    plt.close()
    print(f"Confusion Matrix saved for {dataset_name} - {level_name}")

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

def plot_evaluation_metrics(metrics, output_folder, task_name):
    metric_names = ['Accuracy', 'F1 Score', 'AUC']
    metric_values = [metrics['accuracy'], metrics['f1'], metrics['auc']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    plt.ylabel('Metric Value')
    plt.title(f'Evaluation Metrics for {task_name}')
    plt.ylim(0, 1)
    
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{value:.2f}', ha='center', va='bottom', color='white', fontweight='bold')
    
    plot_file = output_folder / f'{task_name}_evaluation_metrics.png'
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    print(f"Evaluation metrics plot saved as {plot_file}")

def save_evaluation_metrics_to_csv(metrics, output_folder):
    metrics_file = output_folder / 'evaluation_metrics_summary.csv'
    if metrics_file.exists():
        metrics_df = pd.read_csv(metrics_file)
    else:
        metrics_df = pd.DataFrame(columns=['task_name', 'accuracy', 'f1', 'auc'])
    
    metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Evaluation metrics saved to {metrics_file}")

def run_bootstrap_auc():
    input_file = Path(args.input)
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    # mapping to patient and organ information
    mapping_file = '/sc/arion/projects/comppath_SAIF/data/skin/cohort_08_15_2024/MTL110623_master.csv'

    # precalculated results of probabilities in each class
    probs_df = pd.read_csv(input_file)
    probs_df['real'] = probs_df.target.map({0: 'White', 1: 'Black', 2: 'Hispanic/Latino', 3: 'Asian', 4: 'Other'})
    probs_df['pred'] = probs_df[['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']].idxmax(1)
    probs_df['correct'] = probs_df['pred'] == probs_df['real'] 
    probs_df = map_slide_to_patient_and_organ(probs_df, mapping_file)
    
    task_name = 'slide_level'
    task_output_folder = output_folder / task_name
    task_output_folder.mkdir(parents=True, exist_ok=True)
    
    # summary statistics of model performance of race prediction task
    run_bootstrap_for_level(probs_df, task_output_folder, 'slide_level', task_name)
    # find representative examples for each class for visualization
    race_results_df = find_representative_examples_by_race(probs_df, num_samples=5)
    race_results_df.to_csv(task_output_folder / f'{task_name}_examples.csv')

def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_true, y_prob, multi_class='ovr'),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred)
    }
    return metrics

def bootstrap_metrics(y_true, y_pred, y_prob, n_bootstraps=1000):
    bootstrapped_metrics = {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'mcc', 'kappa']}
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        metrics = calculate_metrics(y_true[indices], y_pred[indices], y_prob[indices])
        for metric, value in metrics.items():
            bootstrapped_metrics[metric].append(value)
    
    metrics_summary = {metric: {'mean': np.mean(values), 'std': np.std(values)} for metric, values in bootstrapped_metrics.items()}
    return metrics_summary

def plot_metrics(metrics_summary, output_folder, task_name):
    for metric, values in metrics_summary.items():
        plt.figure()
        plt.bar(['mean'], [values['mean']], yerr=[values['std']], capsize=5)
        plt.title(f'{metric.capitalize()} with 95% CI')
        plt.savefig(output_folder / f'{task_name}_{metric}_95CI.png')
        plt.close()

def run_bootstrap_for_level(probs_df, output_folder, level, task_name, pooling='average'):
    results_file = output_folder / f'{task_name}_bootstrap_results.pkl'
    output_file = output_folder / f'{task_name}_metrics_results.csv'

    plot_confusion_matrix(probs_df, '21M', output_folder, task_name)
    plot_roc_curve(probs_df, '21M', output_folder, task_name)
    
    if results_file.exists():
        print(f"Loading precomputed {task_name} bootstrap results...")
        with open(results_file, 'rb') as f:
            metrics_summary = pickle.load(f)
    else:
        print(f"Calculating {level} metrics with {pooling} pooling...")
        y_true = probs_df['real']
        y_pred = probs_df['pred']
        y_prob = probs_df[['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other']].values
        metrics_summary = bootstrap_metrics(y_true, y_pred, y_prob)
        class_metrics_summary = bootstrap_metrics_by_class(probs_df)
        with open(results_file, 'wb') as f:
            pickle.dump((metrics_summary, class_metrics_summary), f)
    
    metrics_df = pd.DataFrame(metrics_summary).T
    metrics_df.to_csv(output_file, index=True)
    plot_metrics(metrics_summary, output_folder, task_name)
    save_class_metrics_to_csv(class_metrics_summary, output_folder, task_name)
    plot_class_metrics(class_metrics_summary, output_folder, task_name)
    print(f"Metrics saved to {output_file}")

def bootstrap_metrics_by_class(probs_df, n_bootstraps=1000, confidence_level=0.95):
    class_metrics = {}
    for class_name in class_prob_columns:
        class_df = probs_df[probs_df['real'] == class_name].reset_index(drop=True)
        y_true = class_df['real']
        y_pred = class_df['pred']
        y_prob = class_df[class_prob_columns].values
        class_metrics[class_name] = bootstrap_metrics(y_true, y_pred, y_prob, n_bootstraps)
    return class_metrics

def plot_class_metrics(class_metrics, output_folder, task_name):
    for class_name, metrics_summary in class_metrics.items():
        for metric, values in metrics_summary.items():
            plt.figure()
            plt.bar(['mean'], [values['mean']], yerr=[values['std']], capsize=5)
            plt.title(f'{class_name} {metric.capitalize()} with 95% CI')
            plt.savefig(output_folder / f'{task_name}_{class_name}_{metric}_95CI.png')
            plt.close()

def save_class_metrics_to_csv(class_metrics, output_folder, task_name):
    for class_name, metrics_summary in class_metrics.items():
        metrics_df = pd.DataFrame(metrics_summary).T
        metrics_df.to_csv(output_folder / f'{task_name}_{class_name}_metrics_results.csv', index=True)

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