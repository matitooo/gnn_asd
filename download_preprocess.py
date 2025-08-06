import os
import shutil
from nilearn import datasets
from ABIDE_utils import ABIDEParser as Reader
import yaml


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Configuration
pipeline = config["pipeline"]
n_subjects = config["n_subjects"]
atlas_name = config["atlas_name"]
kind = config["kind"]
root_folder = config["root_folder"]

data_folder = os.path.join(root_folder, f"ABIDE_pcp/{pipeline}/filt_noglobal")

filemapping = {"rois_cc200": "rois_cc200.1D"}

# 1. Create data directory
os.makedirs(data_folder, exist_ok=True)
shutil.copyfile(
    "ABIDE_utils/subject_IDs.txt", os.path.join(data_folder, "subject_IDs.txt")
)


# 2. Data download
abide = datasets.fetch_abide_pcp(
    data_dir=root_folder,
    n_subjects=n_subjects,
    pipeline=pipeline,
    band_pass_filtering=True,
    global_signal_regression=False,
    derivatives=["rois_cc200"],
)

# Organizing data per subject
subject_IDs = Reader.get_ids(n_subjects)
subject_IDs = subject_IDs.tolist()

for s, fname in zip(
    subject_IDs, Reader.fetch_filenames(subject_IDs, f"rois_{atlas_name}")
):
    subject_folder = os.path.join(data_folder, s)
    os.makedirs(subject_folder, exist_ok=True)

    base = fname.split(f"rois_{atlas_name}")[0]
    src = base + filemapping[f"rois_{atlas_name}"]
    dst = os.path.join(subject_folder, base + filemapping[f"rois_{atlas_name}"])
    if not os.path.exists(dst):
        shutil.move(src, dst)

# Time series extraction
time_series = Reader.get_timeseries(subject_IDs, atlas_name)


# Connectivity matrix creation
for i, ts in enumerate(time_series):
    subject_id = subject_IDs[i]
    conn = Reader.subject_connectivity(ts, subject_id, atlas_name, kind, save=False)

    save_path = os.path.join(
        data_folder, subject_id, f"{subject_id}_{atlas_name}_{kind}.mat"
    )
    Reader.sio.savemat(save_path, {"connectivity": conn})

folders = [name for name in os.listdir(data_folder)]
folders.sort()

with open("subject_IDs.txt", "w") as f:
    for folder in folders:
        f.write(f"{folder}\n")

print(f"{len(folders)} found!")
print("Preprocessing completed")
