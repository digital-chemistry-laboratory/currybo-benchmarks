import os 
import pickle

datasets = {
    "Cernak": {
        "data": "Cernak_enhanced.csv",
        "conditions": [
            {
                "name": "Catalyst",
                "type": "smiles",
            },
            {
                "name": "Base",
                "type": "smiles",
            }
        ],
        "targets": [
            {
                "name": "Conversion",
                "type": "scalar"
            }
        ],
        "substrates": [
            {
                "name": "Nucleophile",
                "type": "smiles"
            }
        ],
        "objectives": [
            {
                "name": "Conversion",
                "maximize": True,
                "abs_threshold": 0.01
            }       
        ],
        "num_samples": [12]
    },
    "Denmark": {
        "data": "Denmark_enhanced.csv",
        "conditions": [
            {
                "name": "Catalyst",
                "type": "smiles",
            }
        ],
        "substrates": [
            {
                "name": "Imine",
                "type": "smiles",
            },
            {
                "name": "Thiol",
                "type": "smiles",
            },
        ],
        "targets": [
            {
                "name": "Delta_Delta_G",
                "type": "scalar"
            }
        ],
        "objectives": [
            {
                "name": "Delta_Delta_G",
                "maximize": True,
                "abs_threshold": 0.5
            }       
        ],
        "num_samples": [3, 3]
    },
    "DenmarkMOBO": {
        "data": "Denmark_enhanced_MOBO.csv",
        "conditions": [
            {
                "name": "Catalyst", 
                "type": "smiles",
            },
            # {
            #     "name": "input_desc",
            #     "type": "scalar",
            # }
        ],
        "targets": [
            {
                "name": "Delta_Delta_G",
                "type": "scalar",
            },
            {
                "name": "avgMolWt",
                "type": "scalar",
            },
            # {
            #     "name": "input_desc",
            #     "type": "scalar",
            # }
        ],
        "substrates": [
            {
                "name": "Imine", 
                "type": "smiles",
            },
            {
                "name": "Thiol",
                "type": "smiles",
            }
        ],
        "objectives": [
            # {
            #     "name": "input_desc",
            #     "maximize": False,
            #     "abs_threshold": 500,
            #     "maximize": False,
            # },
            {
                "name": "avgMolWt",
                "maximize": True,
                "abs_threshold": 500,
            },
            {
                "name": "Delta_Delta_G",
                "maximize": True,
                "abs_threshold": 0.5,
            },
        ],
        "num_samples": [3, 3]
    },
    "DenmarkASGP": {
        "data": "Denmark_ASGP.csv",
        "conditions": [
            {
                "name": "Catalyst",
                "type": "smiles",
            }
        ],
        "substrates": [
            {
                "name": "Imine Desc",
                "type": "array",
            },
            {
                "name": "Thiol",
                "type": "smiles",
            },
        ],
        "targets": [
            {
                "name": "Delta_Delta_G",
                "type": "scalar"
            }
        ],
        "objectives": [
            {
                "name": "Delta_Delta_G",
                "maximize": True,
                "abs_threshold": 0.5
            }       
        ],
        "num_samples": [3, 3]
    },
    "Doyle": {
        "conditions": ["Ligand", "Additive", "Base"],
        "target": ["Yield"],
        "num_samples": [10]
    },
    "Deoxyfluorination": {
        "data": "deoxyfluorination_enhanced.csv",
        "conditions": [
            {
                "name": "base",
                "type": "smiles",
            },
            {
                "name": "fluoride",
                "type": "smiles",
            }
        ],
        "targets": [
            {
                "name": "yield",
                "type": "scalar"
            }
        ],
        "substrates": [
            {
                "name": "substrate",
                "type": "smiles"
            }
        ],
        "objectives": [
            {
                "name": "yield",
                "maximize": True,
                "abs_threshold": 0.5
            }       
        ],
        "num_samples": [25]
    },
    "Borylation": {
        "data": "borylation_enhanced.csv",
        "conditions": [
            {
                "name": "ligand",
                "type": "smiles",
            },
            {
                "name": "solvent",
                "type": "smiles",
            }
        ],
        "targets": [
            {
                "name": "yield",
                "type": "scalar"
            }
        ],
        "substrates": [
            {
                "name": "Electrophile",
                "type": "smiles"
            }
        ],
        "objectives": [
            {
                "name": "yield",
                "maximize": True,
                "abs_threshold": 0.4
            }       
        ],
        "num_samples": [20]
    }
} 

def load_proxy_model(pickle_path: str):
    """
    Load pickled ML model.

    Args:
        pickle_path (str): Path of the pickled model to load.
    
    Returns:
        model.predict (callable): Callable method to directly perform predictions on.
    """
    with open(pickle_path, 'rb') as file:
        model = pickle.load(file)
    return model.predict
