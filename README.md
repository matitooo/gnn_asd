# A GNN implementation for ASD diagnosys

### <ins>Final Project for Sose25 class GNN at Potsdam Universität </ins>

## How to use 

### 1. Install Required packeges
Required packages  can be installed by running the following command

```
pip install -r requirements.txt
```


### 2. Configuration file editing

 1.1. Rename config.yaml.template file to config.yaml
 1.2. Edit config.yaml by specifying the project root folder (it must be a absolute path)

### 3. Data downloading and preprocessing

Using the download_preprocess.py script, download the [Preprocessed ABIDE Dataset](http://preprocessed-connectomes-project.org/abide/)  with the following command

```
python download_preprocess.py

```
Note: The ABIDE parser script and part of download_preprocess.py  is from [Population_GCN](https://github.com/parisots/population-gcn). 





