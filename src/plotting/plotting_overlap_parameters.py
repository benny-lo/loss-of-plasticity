import zipfile
import pickle
import matplotlib.pyplot as plt
import re
import yaml
from plot_util import save_plot, generate_pruned_string

def load_data_from_zip(zip_file_path, grad_files_in_zip):
    all_data = {}
    with zipfile.ZipFile(zip_file_path, 'r') as zf:
        for grad_file in grad_files_in_zip:
            if grad_file in zf.namelist():
                with zf.open(grad_file) as file:
                    data = pickle.load(file)
                    all_data[grad_file] = data
                #print(f"Data loaded successfully for {grad_file}")
            else:
                print(f"File '{grad_file}' not found in the zip archive.")
    return all_data

def plot_data(all_data, save_path):
    plt.figure(figsize=(10, 6))
    for grad_file, data in all_data.items():
        # Extract keys and values
        keys = [int(k) for k in data.keys()]  # Ensure keys are integers
        values = [tensor.item() for tensor in data.values()]  # Extract scalar values

        # Generate a label using the pruning percentage and rounds
        label = generate_pruned_string(grad_file) or grad_file.split('/')[-1]

        # Plot the data
        plt.plot(keys, values, marker='o', label=label)

    plt.xlabel('Task')
    plt.ylabel('Parameter Overlap')
    plt.title('Parameter Overlap Across Versions')
    plt.legend()
    plt.grid(True)

    # Save the plot
    save_plot(plt.gcf(), save_path)
    print(f"Plot saved to {save_path}")


def main():
    config_path = "plot_config.yaml"

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    zip_file_plasticity_exp_path = config['zip_file_plasticity_exp_path']
    grad_files_in_zip = config['grad_files_in_zip']
    plot_path = config['imp_plot_path']

    all_data = load_data_from_zip(zip_file_plasticity_exp_path, grad_files_in_zip)

    plot_data(all_data, plot_path)


if __name__ == '__main__':
    main()
