# AntioxidantActivity
Antioxidant Activity predction IC50


create the AntioxidantDPPH env
Dependencies ...


To run:

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