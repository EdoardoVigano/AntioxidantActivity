import pandas as pd
import pickle
import os
from mordred import Calculator, descriptors, error

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
import numpy as np
import argparse
import sys
from datetime import datetime


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
    with open(os.path.join(os.getcwd(), 'models', 'alldata_model_Antioxidant_DPPH30MIN_extra_trees_regressor.pkl'), 'rb') as f:
        model1 = pickle.load(f)

    with open(os.path.join(os.getcwd(),'models', 'alldata_model_Antioxidant_DPPH30MIN_xgb_regressor.pkl'), 'rb') as f:
        model2 = pickle.load(f)
    
    with open(os.path.join(os.getcwd(),'models', 'alldata_model_Antioxidant_DPPH30MIN_gradient_boosting_regressor.pkl'), 'rb') as f:
        model3 = pickle.load(f)

    with open(os.path.join(os.getcwd(), 'pipeline_and_AD', 'pipeline_Antioxidant_DPPH30MIN.pkl'), 'rb') as f:
        pipeline = pickle.load(f)

    with open(os.path.join(os.getcwd(), 'pipeline_and_AD', 'AD_clf.pkl'), 'rb') as f:
        ad = pickle.load(f)
        
    return model1, model2, model3, pipeline, ad

def files_importer():
    parser = argparse.ArgumentParser(description='Antioxidant assesment')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smiles', required=False, help='Specify target SMILES')
    group.add_argument('--filename', required=False, help='Specify xlsx file with molecules to predict')
    parser.add_argument('--summary', default=None, help='Specify if you want prediction from all models or only the consensus (default: summary=None) ones')
    
    args = parser.parse_args()
    
    # acess the input value
    input_value = args.smiles
    if input_value is None:
        input_value = args.filename
        
    print(f"Input value: {input_value}")
    return input_value, args.summary


if __name__ == '__main__':
    print("Antioxidant Model:\nExtra trees model to predict IC50 (log(ug/ml))")
    input_value, summary = files_importer()
    
    if ".xlsx" not in input_value: 
        df = pd.DataFrame([input_value], columns=['SMILES'])
    else:
        df = pd.read_excel(input_value)
        
    # import data for testing
    model1, model2, model3, pipeline, ad = pipeline_model_importer()
    # Calculate molecular descriptors
    print("Molecular descriptors calculation....\n")
    data = mordred_calculator(df)
    print("MDs tranformation...\n")
    data_input = pd.DataFrame(pipeline.transform(data.loc[:, pipeline.feature_names_in_]), columns = pipeline.feature_names_in_)
    
    print("Model Assessment...")
    mw_calc = [MolWt(Chem.MolFromSmiles(smi)) for smi in df['SMILES']]
    df['Predictions_ETR [-log(IC50) M]'] = model1.predict(data_input.loc[:, model1.feature_names_in_])
    df['Predictions_ETR [mg/L]'] = [(10**-(c))*mw_*1000 for c, mw_ in zip(df['Predictions_ETR [-log(IC50) M]'], mw_calc)]

    df['Predictions_XGB [-log(IC50) M]'] = model2.predict(data_input.loc[:, model2.feature_names_in_])
    df['Predictions_XGB [mg/L]'] = [(10**-(c))*mw_*1000 for c, mw_ in zip(df['Predictions_XGB [-log(IC50) M]'], mw_calc)]

    df['Predictions_GB [-log(IC50) M]'] = model3.predict(data_input.loc[:, model3.feature_names_in_])
    df['Predictions_GB [mg/L]'] = [(10**-(c))*mw_*1000 for c, mw_ in zip(df['Predictions_GB [-log(IC50) M]'], mw_calc)]

    
    df['Consensus [-log(IC50) M]'] = [np.mean([y1, y2, y3]) for y1, y2, y3 in zip(df['Predictions_ETR [-log(IC50) M]'], df['Predictions_XGB [-log(IC50) M]'], df['Predictions_GB [-log(IC50) M]'])]
    df['Interval [-log(IC50) M]'] = [np.std([y1, y2, y3]) for y1, y2, y3 in zip(df['Predictions_ETR [-log(IC50) M]'], df['Predictions_XGB [-log(IC50) M]'], df['Predictions_GB [-log(IC50) M]'])]
    
    df['Consensus [mg/L]'] = [np.mean([y1, y2, y3]) for y1, y2, y3 in zip(df['Predictions_ETR [mg/L]'], df['Predictions_XGB [mg/L]'], df['Predictions_GB [mg/L]'])]
    df['Interval [mg/L]'] = [np.std([y1, y2, y3]) for y1, y2, y3 in zip(df['Predictions_ETR [mg/L]'], df['Predictions_XGB [mg/L]'], df['Predictions_GB [mg/L]'])]

    
    df = round(df, 3)
    
    df['Consensus AND Uncertanty [mg/L]'] = [f"{str(y)} \u00B1 {i}" for y, i in zip(df['Consensus [mg/L]'], df['Interval [mg/L]'])] 
    df['Applicability Domain'] = ad.predict(data_input.loc[:, model2.feature_names_in_])
    current_date = datetime.now().strftime("%d_%m_%Y")# ("%Y-%m-%d")
    if summary:
        df.loc[:, ['Consensus AND Uncertanty [mg/L]', 'Applicability Domain']].to_excel(f'Summary_predictions_{current_date}.xlsx')
        print(df)
    else:
        df.to_excel(f'Predictions_{current_date}.xlsx')
        print(df)
    
    
    
    
        
        

        