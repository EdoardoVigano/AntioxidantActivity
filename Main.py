import pandas as pd
import pickle
import os
from mordred import Calculator, descriptors, error

from rdkit import Chem
import numpy as np
import argparse
import sys



def mordred_calculator(dataset:pd.DataFrame):
    """
    Args:
        dataset (pd.DataFrame): the dataset must be provided with "SMILES" column for which the MDs will be calculated

    Returns:
        The original dataset with concatenated the MDs
    """
    
    mols = [Chem.MolFromSmiles(smi) for smi in dataset['SMILES']]
    calc = Calculator(descriptors, ignore_3D=True)
    # as pandas
    df = calc.pandas(mols)
    df = pd.concat([dataset, df], axis=1)

    df = df.applymap(lambda x: np.nan if isinstance(x, error.Error) or isinstance(x, error.Missing)else x) # remove errors using nan
    return df

def check_smiles(smiles):
    try:
        Chem.MolFromSmiles(smiles)
    except:
        print(f"{smiles}: invalid smiles!")
        sys.exit(1)

def pipeline_model_importer():
    
    # import model and pipeline
    with open(os.path.join(os.getcwd(), 'models', 'model_extratrees.pkl'), 'rb') as f:
        model1 = pickle.load(f)

    with open(os.path.join(os.getcwd(),'models', 'model_xgb.pkl'), 'rb') as f:
        model2 = pickle.load(f)

    with open(os.path.join(os.getcwd(), 'pipeline_and_AD', 'pipeline.pkl'), 'rb') as f:
        pipeline = pickle.load(f)

    with open(os.path.join(os.getcwd(), 'pipeline_and_AD', 'AD_clf.pkl'), 'rb') as f:
        ad = pickle.load(f)
        
    return model1, model2, pipeline, ad

def files_importer():
    parser = argparse.ArgumentParser(description='Antioxidant assesment')
    parser.add_argument('--smiles', required=False, help='Specify target SMILES')
    parser.add_argument('--filename', required=False, help='Specify xlsx file with molecules to predict')
    
    args = parser.parse_args()
    
    # acess the input value
    input_value = args.smiles
    if input_value is None:
        input_value = args.filename
        
    print(f"Input value: {input_value}")
    return input_value


if __name__ == '__main__':
    print("Antioxidant Model:\nExtra trees model to predict IC50 (log(ug/ml))")
    input_value = files_importer()
    
    if ".xlsx" not in input_value: 
        df = pd.DataFrame([input_value], columns=['SMILES'])
    else:
        df = pd.read_excel(input_value)
        
    # import data for testing
    model1, model2, pipeline, ad = pipeline_model_importer()
    # Calculate molecular descriptors
    print("Molecular descriptors calculation....\n")
    data = mordred_calculator(df)
    print("MDs tranformation...\n")
    data_input = pd.DataFrame(pipeline.transform(data.loc[:, pipeline.feature_names_in_]), columns = pipeline.feature_names_in_)
    
    
    print("Model Assessment...")

    df['Predictions_extratrees'] = model1.predict(data_input.loc[:, model1.feature_names_in_])
    df['Predictions_XGB'] = model2.predict(data_input.loc[:, model2.feature_names_in_])
    df['Applicability Domain'] = ad.predict(data_input.loc[:, model2.feature_names_in_])
    df.to_excel('predictions.xlsx')
    print(df)
    
    
    
        
        

        