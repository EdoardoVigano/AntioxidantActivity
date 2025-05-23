# Antioxidant Activity: In silico models for predicting the antioxidant activity of small molecules relevant to human health
Our software has been developed for predicting the antioxidant activity of small molecules (< 1000 Da) and it aims to assist in identifying potential substances that could be applied in health support. It is built on regression models developed on an expert-curated dataset of antioxidants.
Given the SMILES as input the software will predict the half-maximal inhibitory concentration (IC50) of the substance(s) of interest.
## Installation
1.	Download from [here](https://github.com/EdoardoVigano/AntioxidantActivity) the AntioxidantActivity_DPPH folder and unzip it
2.	Do not move the files out of the folder.
3.	Install Anaconda prompt following the instruction [here](https://www.anaconda.com/download)
4.	Open Anaconda terminal and create the AntioxidantDPPH environment with the command:
            > conda create --name AntioxidantActivity_DPPH python=3.11
5.	Activate your environment with the command:
6.	        > conda activate AntioxidantActivity_DPPH
7.	Install the following dependencies with the command:
8.	        > pip install scikit-learn==1.4.0 rdkit==2023.9.4 pandas==2.2.0 mordred==1.2.0 xgboost==2.1.3
9.	Move to the AntioxidantActivity_DPPH folder before prompt the target compound
## Usage
To run the program:
### Single molecule mode:
Command -> python Main.py --smiles [write single SMILES] [optional]: --summary 1

### Batch mode:
Command -> python Main.py--file [add file name] [optional]: --summary 1
Key:   --file: path of file to predict the antioxidant activity must have column named SMILES (batch functionality)
### OPTIONAL: set summary to one
        --summary 1 to obtain only the consensus prediction and uncertanty value.
        [default] --summary None to obtain all the models' predictions.

### Examples:
    python Main.py --smiles c1ccccc1CCN --summary 1
    python Main.py --file test.xlsx --summary 1 
    python Main.py --file test.xlsx 
