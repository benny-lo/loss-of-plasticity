import zipfile
import pickle
import matplotlib.pyplot as plt
import re
import numpy as np
import yaml
import csv
from plot_util import extract_pickle_from_zip, save_plot, generate_pruned_string

import scienceplots  # Import the scienceplots for publication quality
plt.style.use(['science', 'ieee'])

def compute_differences(data):
    return np.diff(data)


def compute_correlation(overlap, test_acc, normalize=False):
    if normalize:
        overlap = (overlap - np.mean(overlap)) / np.std(overlap)
        test_acc = (test_acc - np.mean(test_acc)) / np.std(test_acc)
    return np.corrcoef(overlap, test_acc)[0, 1]


def preprocess_data(data):
    tasks = sorted(map(int, data.keys()))
    avg_overlap = [data[str(task)]["average pairwise overlap"] / 100 for task in tasks]
    overlap_std = [data[str(task)]["pairwise overlap std"] for task in tasks]
    tasks = tasks[1:]
    avg_overlap = avg_overlap[1:]
    return tasks, avg_overlap, overlap_std

def moving_average(signal, window_size):
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    padded_signal = np.pad(signal, (half_window, half_window), mode='edge')
    return np.convolve(padded_signal, np.ones(window_size) / window_size, mode='valid')

def moving_variance(signal, window_size):
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    padded_signal = np.pad(signal, (half_window, half_window), mode='edge')
    smoothed_variance = [
        np.var(padded_signal[i:i+window_size])
        for i in range(len(signal))
    ]
    return np.array(smoothed_variance)

def get_test_accuracies_at_tasks(task_performance, tasks):
    test_acc = task_performance.get("test_acc", [])
    test_acc_at_tasks = [
        test_acc[task - 1] if task - 1 < len(test_acc) else 0.0
        for task in tasks
    ]
    return test_acc_at_tasks


def plot_overlap_and_accuracy(tasks, avg_overlap_dict, test_acc, window_size, save_path=None):
    smoothed_acc = moving_average(test_acc, window_size)
    fig, ax1 = plt.subplots(figsize=(8, 6))

    for label, avg_overlap in avg_overlap_dict.items():
        ax1.plot(tasks, avg_overlap, marker="o", label=f"Average Pairwise Overlap ({label})")

    ax1.set_xlabel("Tasks", fontsize=12)
    ax1.set_ylabel("Average Pairwise Overlap", color="blue", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(tasks[:len(test_acc)], test_acc, color="orange", label="Test Accuracy at Tasks", zorder=5)
    ax2.set_ylabel("Test Accuracy", color="green", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="green")

    plt.title("Average Pairwise Overlap and Test Accuracy at Tasks", fontsize=14)
    fig.tight_layout()

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    

    if save_path:
        fig.savefig(save_path)
        
    plt.show()

def plot_differenced_overlap_and_accuracy(tasks, avg_overlap_dict, test_acc, window_size, save_path=None, correlation_csv_path=None):
    diff_test_acc = compute_differences(test_acc)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    correlations = []  # To store the correlations and their labels

    for label, avg_overlap in avg_overlap_dict.items():
        diff_overlap = compute_differences(avg_overlap)

        correlation = compute_correlation(avg_overlap, test_acc)
        diff_correlation = compute_correlation(diff_overlap, diff_test_acc)

        print(f"Pearson Correlation between {label} Overlap and Test Accuracy: {correlation:.4f}")
        print(f"Pearson Correlation between Differenced {label} Overlap and Test Accuracy: {diff_correlation:.4f}")

        # Store the correlation and label for CSV output
        correlations.append([label, correlation, diff_correlation])

        ax1.plot(tasks[1:], diff_overlap, marker="o", label=f"Differenced {label} Pairwise Overlap")

    ax2.plot(tasks[1:], diff_test_acc, marker="o", label="Differenced Test Accuracy", color="green")

    ax1.set_ylabel("Differenced Average Pairwise Overlap", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.legend(loc="upper right")

    ax2.set_xlabel("Tasks")
    ax2.set_ylabel("Differenced Test Accuracy", color="green")
    ax2.tick_params(axis='y', labelcolor="green")
    ax2.legend(loc="upper right")

    plt.suptitle("Differenced Average Pairwise Overlap and Test Accuracy", fontsize=14)
    fig.tight_layout()
    

    if save_path:
        save_plot(fig, save_path)
    
    plt.show()

    # Save the correlation results to CSV if the path is provided
    if correlation_csv_path:
        with open(correlation_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Label', 'Correlation (Original)', 'Correlation (Differenced)'])  # Header
            writer.writerows(correlations)
        print(f"Correlations have been saved to {correlation_csv_path}.")


def plot_smoothed_test_accuracy(tasks, test_acc, window_size, save_path=None):
    smoothed_test_acc = moving_average(test_acc, window_size)
    smoothed_variance = moving_variance(test_acc, window_size)
    smoothed_std_dev = np.sqrt(smoothed_variance)

    smoothed_tasks = tasks[:len(smoothed_test_acc)]

    plt.figure(figsize=(8, 6))
    plt.plot(smoothed_tasks, smoothed_test_acc, label=f"Smoothed Test Accuracy (Window={window_size})", color="blue")

    lower_bound = smoothed_test_acc - smoothed_std_dev
    upper_bound = smoothed_test_acc + smoothed_std_dev
    plt.fill_between(smoothed_tasks, lower_bound, upper_bound, color="blue", alpha=0.2, label="Confidence Region (\u00b11 Std. Dev.)")

    plt.xlabel("Tasks", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("Smoothed Test Accuracy with Confidence Region Over Tasks", fontsize=14)
    plt.legend()
    plt.tight_layout()
    

    if save_path:
        plt.savefig(save_path)
        
    plt.show()

def process_and_plot_data(config):
    avg_overlap_dict = {}

    for pickle_file_name in config["winning_tickets_overlap_files"]:
        data = extract_pickle_from_zip(
            config["zip_file_path"], pickle_file_name, config["output_file_path"]
        )
        tasks, avg_overlap, _ = preprocess_data(data)
        avg_overlap_dict[generate_pruned_string(pickle_file_name)] = avg_overlap

    task_performance = extract_pickle_from_zip(
        config["results_zip_path"], config["task_performance_file"], config["output_file_path"]
    )
    test_acc = get_test_accuracies_at_tasks(task_performance, tasks)

    plot_overlap_and_accuracy(
        tasks, avg_overlap_dict, test_acc, config["window_size"], save_path=config.get("overlap_accuracy_plot_path")
    )
    plot_differenced_overlap_and_accuracy(
        tasks, avg_overlap_dict, test_acc, config["window_size"], 
        save_path=config.get("differenced_plot_path"),
        correlation_csv_path=config["correlation_save_path"]
    )

    if "train_acc" in task_performance:
        plot_smoothed_test_accuracy(
            range(len(task_performance["train_acc"])),
            task_performance["train_acc"],
            config["window_size"],
            save_path=config["smoothed_test_accuracy_plot_path"]
        )
def main():
    # Load configuration from the YAML file
    config_path = "plot_config.yaml"  # Assuming the config file is named 'config.yaml'
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    # Call the data processing and plotting function with the loaded config
    process_and_plot_data(config)

if __name__ == "__main__":
    main()