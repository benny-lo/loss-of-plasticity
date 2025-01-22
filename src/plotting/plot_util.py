import zipfile
import pickle
import matplotlib.pyplot as plt
import re

def save_plot(fig, filename):
    fig.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    
def extract_pickle_from_zip(zip_file_path, pickle_file_name, output_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        if pickle_file_name in zip_ref.namelist():
            with zip_ref.open(pickle_file_name) as file:
                data = pickle.load(file)
            with open(output_file_path, 'wb') as output_file:
                pickle.dump(data, output_file)
            return data
        else:
            raise FileNotFoundError(f"File {pickle_file_name} not found in the ZIP archive.")

def generate_pruned_string(name):
    match = re.search(r"target_percentage_([\d.]+)_pruning_rounds_([\d.]+)", name, re.IGNORECASE)
    if match:
        y, x = match.groups()
        return f"Prun. Perc.: {y}, Rds.: {x}"
    return None