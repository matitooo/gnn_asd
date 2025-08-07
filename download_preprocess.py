import yaml
from utils import abide_download, id_list_creation

# Extract yaml config file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract parameters from config file
root_folder=config["root_folder"]
pipeline=config["pipeline"]
n_subjects=config["n_subjects"]
data_folder = config["data_folder"]
phenotype_file = config["phenotype_file"]
subject_ids = config["subject_ids_file"]
atlas_name = config["atlas_name"]
kind = config["kind"]

abide_download(root_folder,pipeline,n_subjects,atlas_name,kind)
id_list_creation(root_folder,pipeline)