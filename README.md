# Antioxidant Activity: In silico models for predicting the antioxidant activity of small molecules relevant to human health
Our software has been developed for predicting the antioxidant activity of small molecules (< 1000 Da) and it aims to assist in identifying potential substances that could be applied in health support. It is built on regression models developed on an expert-curated dataset of antioxidants.
Given the SMILES as input the software will predict the half-maximal inhibitory concentration (IC50) of the substance(s) of interest.
## Installation
1.	Download from [here](https://github.com/EdoardoVigano/AntioxidantActivity) the AntioxidantActivity_DPPH folder and unzip it
2.	Do not move the files out of the folder
3.	Create the AntioxidantDPPH environment
4.	Using python 3.11.10 install the following dependencies: scikit-learn==1.4.0 rdkit==2023.9.4 pandas==2.2.0 mordred==1.2.0 xgboost==2.1.3
## Usage
To run the program:

1. Command:
    python Main.py --file [add file name] or --smiles [write single SMILES] [optional]: --summary 1

    key: 
        --file: path of file to predict the antioxidant activity must have column named SMILES (batch functionality)
        OR
        --smiles: write single SMILES to predic (single molecule functionality)

        OPTIONAL: set summary to one
        --summary 1 to have only the consensus prediction and uncertanty value.
        [default] --summary None all model predictions are reported


2. examples:
    python Main.py --smiles c1ccccc1CCN --summary 1
    python Main.py --file test.xlsx --summary 1
    python Main.py --file test.xlsx 
