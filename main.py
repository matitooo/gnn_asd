import yaml
import torch
from graph_creation_utils import matrix_preprocessing, graph_creation
from utils import data_conversion,data_split,evaluate
from model_utils import train, test
from model import ASDGCN
from sklearn.preprocessing import StandardScaler
import wandb

def sweep():
    wandb.init()
    config=wandb.config

    lr=config.lr
    hidden_size=config.hidden_size
    n_epochs=config.epochs

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

    #Data object creation
    data=data_conversion(features,y,adjacency)
    scaler = StandardScaler()
    data.x = torch.tensor(scaler.fit_transform(data.x), dtype=torch.float)

    #Train/Test split
    idx_train,idx_test=data_split(y,test_ratio=0.2)

    model = ASDGCN(num_features=features.shape[1], hidden_channels=hidden_size, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs+1):
        loss = train(model,optimizer,criterion,data,idx_train)
        acc, f_1 = evaluate(model, data, idx_test)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test Acc: {acc:.4f} | Test F1: {f_1:.4f}")

    wandb.log({'Accuracy':acc})
    wandb.log({'F_1 Score': f_1})
    wandb.log({'Loss':loss})
    

    # out = model(data)
    # print(out.min().item(), out.max().item(), out.mean().item())

if __name__ == "__main__":
    sweep()