# Antioxidant Activity: In silico models for predicting the antioxidant activity of small molecules relevant to human health
Our software has been developed for predicting the antioxidant activity of small molecules (< 1000 Da) and it aims to assist in identifying potential substances that could be applied in health support. It is built on regression models developed on an expert-curated dataset of antioxidants.
Given the SMILES as input the software will predict the half-maximal inhibitory concentration (IC50) of the substance(s) of interest.
## Installation
1.	Download from [here](https://github.com/EdoardoVigano/AntioxidantActivity) the AntioxidantActivity_DPPH folder and unzip it
2.	Do not move the files out of the folder.
3.	Create the AntioxidantDPPH environment: conda create --name AntioxidantActivity_DPPH python=3.11
4.	pip install the following dependencies: scikit-learn==1.4.0 rdkit==2023.9.4 pandas==2.2.0 mordred==1.2.0 xgboost==2.1.3
5.	To use by terminal: conda activate AntioxidantActivity and open the folder "AntioxidantActivity_DPPH"
## Usage
To run the program:
### Single molecule mode:
Command -> python Main.py --smiles [write single SMILES] [optional]: --summary 1

###Batch mode:
