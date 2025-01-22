import pickle
import matplotlib.pyplot as plt
import yaml
from plot_util import extract_pickle_from_zip, get_test_accuracies_at_tasks

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_task_performance_data(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

def plot_task_performance(config):
    task_performance = extract_pickle_from_zip(
        config["results_zip_path"],
        config["task_performance_file"],
        config["output_file_path"]
    )
    task_performance_data = load_task_performance_data(config["win_tick_task_performance_file_path"])
    test_acc = get_test_accuracies_at_tasks(task_performance, range(1, 101))

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 101), test_acc, color='green', linestyle='-', linewidth=2,
             marker='o', markersize=4, label='Full Network Test Accuracy')
    plt.plot(range(len(task_performance_data.get("test_acc", []))), 
             task_performance_data.get("test_acc", []), 
             color='blue', linestyle='--', linewidth=2, 
             marker='s', markersize=4, label='Winning Ticket Test Accuracy')
    plt.xlabel('Tasks', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Task Performance Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    output_path = config["win_tick_LOP_plot_out"]
    plt.savefig(output_path, format='png')
    
    plt.show()

def main(config_path="plot_config.yaml"):
    try:
        config = load_config(config_path)
        plot_task_performance(config)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
