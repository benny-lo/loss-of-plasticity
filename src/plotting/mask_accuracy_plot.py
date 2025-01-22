import yaml
import matplotlib.pyplot as plt
import numpy as np
from plot_util import extract_pickle_from_zip, get_test_accuracies_at_tasks, save_plot


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def process_data(data_dict):
    #We have accuracies for several masks, this function computes the mean
    tasks = []
    mean_accuracies = []
    for task, values in data_dict.items():
        tasks.append(int(task))
        test_acc_list = values.get('test_acc', [])
        if test_acc_list:
            mean_accuracies.append(np.mean(test_acc_list))
        else:
            mean_accuracies.append(0)
    return tasks, mean_accuracies


def main(config_path="plot_config.yaml"):
    config = load_config(config_path)
    files = config['mask_accuracy_files']

    data = extract_pickle_from_zip(
        files['zip_file_path'], files['random_masks_pickle'], files['output_file_path'])
    tasks1, mean_test_acc1 = process_data(data)

    winning_tickets_data = extract_pickle_from_zip(
        files['zip_file_path'], files['winning_tickets_pickle'], files['output_file_path'])
    tasks2, mean_test_acc2 = process_data(winning_tickets_data)

    task_performance = extract_pickle_from_zip(
        config["results_zip_path"], config["task_performance_file"], config["output_file_path"]
    )

    tasks = sorted(map(int, data.keys()))

    test_acc = get_test_accuracies_at_tasks(task_performance, tasks)
    plt.figure(figsize=(8, 6))
    plt.plot(tasks1, test_acc, color='green', linestyle='-', linewidth=2,
             marker='^', markersize=8, label='Full Network Test Accuracy')
    plt.plot(tasks1, mean_test_acc1, color='blue', linestyle='-', linewidth=2,
             marker='o', markersize=8, label='Random Masks Test Accuracy')
    plt.plot(tasks2, mean_test_acc2, color='red', linestyle='--', linewidth=2,
             marker='s', markersize=8, label='Winning Tickets Test Accuracy')
    plt.xlabel("Task", fontsize=14)
    plt.ylabel("Mean Test Accuracy", fontsize=14)
    plt.title("Mean Test Accuracy Comparison", fontsize=16)
    plt.grid(False)
    plt.gca().set_facecolor('#f9f9f9')
    plt.legend(fontsize=12, loc='best')
    save_plot(plt, files["plot_output_dir"])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
