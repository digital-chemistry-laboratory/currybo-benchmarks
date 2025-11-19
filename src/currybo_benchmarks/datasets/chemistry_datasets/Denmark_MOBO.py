import pandas as pd
from typing import List
from ..discrete_problem import ChemistryDatasetLoader

class DenmarkMOBOLoader(ChemistryDatasetLoader):
    """
    Dataset loader for the Denmark dataset.
    """
    def __init__(self, variable_names: List[str], target: List[str], enhance: bool = True, prefix: str = ""):
        super().__init__(dataset_name="Denmark")
        self.model_path = [prefix + f'src/genbo/test_functions/chemistry_datasets/Denmark_{obj}.pkl' for obj in target]
        if enhance:
            self.csv_path = prefix + 'src/genbo/test_functions/chemistry_datasets/Denmark_enhanced_MOBO.csv'
        else:
            raise ValueError("Denmark MOBO only exists as an enhanced dataset")

        self.load_data()
        self.preprocess_data(variable_names, target)
        self.enhanced = enhance
        self.target = target

    def load_data(self):
        self.dataset = pd.read_csv(self.csv_path, index_col = False)

    def preprocess_data(self, variable_names: List[str], target: str):
        if not set(variable_names) <= set(["Catalyst", "input_desc"]):
            raise ValueError("Invalid variables for dataset")

        unique_smiles = self.dataset["Catalyst"].unique().tolist()
        fp_list = [self.smiles_to_fingerprint(smiles) for smiles in unique_smiles]
        self.x_options["Catalyst"] = fp_list

        if not set(target) <= set(["Delta_Delta_G", "output_desc", "input_desc"]):
            raise ValueError("Invalid targets for dataset")

        w_columns = [col for col in self.dataset.columns if col not in (variable_names + [target]) and col != self.dataset.index.name]

        for column in w_columns:
            unique_smiles = self.dataset[column].unique().tolist()
            fp_list = [self.smiles_to_fingerprint(smiles) for smiles in unique_smiles]
            self.w_options[column] = fp_list
