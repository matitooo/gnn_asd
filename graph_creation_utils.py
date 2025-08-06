import numpy as np
import scipy.io as sio
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def matrix_preprocessing(data_folder,phenotype_file,subject_ids_file,atlas_name,kind):

    subject_ids = np.genfromtxt(subject_ids_file, dtype=str)
    
    # Load connectivity matrixes

    features = []
    eps = 1e-5
    for sid in subject_ids:
        try:
            mat_file = os.path.join(data_folder, str(sid), f"{sid}_{atlas_name}_{kind}.mat")
            mat = sio.loadmat(mat_file)["connectivity"]
            mat = np.clip(mat, -1 + eps, 1 - eps)
            mat = np.arctanh(mat)  # Fisher z-transform
            upper_tri = mat[np.triu_indices_from(mat, k=1)]
            features.append(upper_tri)
        except Exception as e:
            print(f"Error on subject {sid}: {e}")
    
    pheno_df = pd.read_csv(phenotype_file)
    pheno_df = pheno_df[pheno_df["SUB_ID"].isin(subject_ids.astype(int))]

    # Label encoding
    label_encoder = LabelEncoder()
    labels = []

    for sid in subject_ids:
        diagnosis = pheno_df.loc[pheno_df["SUB_ID"] == sid, "DX_GROUP"].values[0]
        labels.append(diagnosis)

    y = label_encoder.fit_transform(labels)  # 1 = ASD, 0 = Control
    return features,y



def graph_creation(data_folder,phenotype_file,subject_ids_file,atlas_name,kind):
    """
    Create adjacency matrix given a subject ID list from the ABIDE dataset. 
    Current phenotipes used for correlation are:
        -Age
        -Sex
    """

    # List of subjects loading
    subject_ids = np.genfromtxt(subject_ids_file, dtype=str).astype(int)

    # Phenotipes loading

    pheno_df = pd.read_csv(phenotype_file)
    pheno_df = pheno_df[pheno_df["SUB_ID"].isin(subject_ids.astype(int))]

    # Label encoding
    label_encoder = LabelEncoder()
    labels = []

    for sid in subject_ids:
        diagnosis = pheno_df.loc[pheno_df["SUB_ID"] == sid, "DX_GROUP"].values[0]
        labels.append(diagnosis)

    y = label_encoder.fit_transform(labels)  # 1 = ASD, 0 = Control
    print(f"Labels shape: {y.shape}")

    """
    Phenotipic features extraction and graph creation

    Features used
    - Age
    - Sex

    Features are normalized and cosine similarity is computed
    """

    #Features array creation
    ages = []
    sexes = []

    for sid in subject_ids:
        row = pheno_df[pheno_df["SUB_ID"] == sid].iloc[0]
        ages.append(row["AGE_AT_SCAN"])
        sexes.append(row["SEX"])  # 1 = Male, 2 = Female

    ages = np.array(ages).reshape(-1, 1)
    sexes = np.array(sexes).reshape(-1, 1)

    # Age normalization
    scaler = StandardScaler()
    ages_norm = scaler.fit_transform(ages)

    # Converting sex into binary values
    sex_bin = (sexes == 1).astype(int)

    # Combined vector creation
    phenotype_features = np.hstack([ages_norm, sex_bin])

    # Similarity matrix computation
    sim_matrix = cosine_similarity(phenotype_features)

    # Using treshold and computing adjacency
    threshold = 0.8
    adjacency = (sim_matrix >= threshold).astype(int)

    # Excluding self-loop
    np.fill_diagonal(adjacency, 0)

    print(f"Population graph adjacency matrix shape: {adjacency.shape}")
    print(f"Number of edges: {adjacency.sum() // 2}")

    """
    GRAPH VISUALIZATION (OPTIONAL)
    """

    # G = nx.from_numpy_array(adjacency)

    # plt.figure(figsize=(10, 8))
    # pos = nx.spring_layout(G, seed=42)

    # """
    # Graph Legend
    # - Red = ASD
    # - Blue = Control
    # """
    # nx.draw_networkx_nodes(
    #     G, pos, node_size=50, node_color=y, cmap=plt.cm.Set1, label="ASD=Red, Control=Blue"
    # )

    # nx.draw_networkx_edges(G, pos, alpha=0.3)

    # plt.title("Population Graph based on phenotipes: age and sex")
    # plt.axis("off")
    # plt.show()
    
    return adjacency