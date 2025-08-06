import yaml
from graph_creation_utils import matrix_preprocessing, graph_creation


# Extract yaml config file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract parameters from config file
data_folder = config["data_folder"]
phenotype_file = config["phenotype_file"]
subject_ids = config["subject_ids_file"]
atlas_name = config["atlas_name"]
kind = config["kind"]

# Matrixes preprocessing
features, y = matrix_preprocessing(
    data_folder, phenotype_file, subject_ids, atlas_name, kind
)

# Graph Creation
adjacency = graph_creation(data_folder, phenotype_file, subject_ids, atlas_name, kind)
