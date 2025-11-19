import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import torch
import importlib.resources as resources

from rdkit import Chem
from rdkit import DataStructs
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from typing import List, Dict

class ChemistryDatasetLoader:
    """
        Parent class for discrete chemistry dataset loaders.
    """
    def __init__(self):
        self.dataset_name = ""
        self.model_path = None
        self.dataset = None
        self.x_options = {}
        self.w_options = {}
        self.min_value = None
        self.max_value = None
        self.continuous_bounds = None

    def smiles_to_fingerprint(self, smiles: str, radius: int=2, n_bits: int=1024):
        """
        Convert SMILES to fingerprints.

        Args:
            smiles (str): SMILES string to convert.
            radius (int): Radius of the Morgan fingerprint.
            n_bits (int): Length of the Morgan fingerprint.

        Returns:
            torch.tensor: Bit fingerprint as torch tensor.
        """
        if smiles is np.nan:
            return torch.zeros(n_bits, dtype=torch.float64)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(n_bits, dtype=torch.float64)
        #fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = mfpgen.GetFingerprint(mol)
        return torch.tensor(fp, dtype=torch.float64)

    def load_model(self, pickle_path: str):
        """
        Load pickled ML model.

        Args:
            pickle_path (str): Path of the pickled model to load.
        
        Returns:
            model.predict (callable): Callable method to directly perform predictions on.
        """
        with open(pickle_path, 'rb') as file:
            model = pickle.load(file)
        return model

    def load_data(self, data_path: str):
        """
        Read csv from dataset.
        """
        filename = data_path.split("/")[-1]
        self.dataset_name = re.sub(r"\..+", "", filename)
        with resources.files("currybo_benchmarks.datasets.chemistry_datasets").joinpath(data_path).open('r') as f:
            self.dataset = pd.read_csv(f, index_col = False)

    def get_samples(self, samples: List[int], cli_args: Dict):
        np.random.seed(1)
        for col, sample in zip(cli_args['substrates'], samples):
            options = pd.unique(self.dataset[col['name']])
            s = np.random.choice(options, sample)
            self.dataset = self.dataset[self.dataset[col['name']].isin(s)]

        return


    def preprocess_data(self, variable_names: List[Dict[str, str]], targets: List[Dict[str, str]], w_columns: List[Dict[str, str]]):
        """
        Process data from csv into options dictionary.

        Args:
            variable_names (List[Dict[str, str]]): List of names of variable in the optimization domain. Example input: [{ "name": "Catalyst", "type": "smiles" }, { "name": "Temperature", "type": "scalar" }]
            targets (List[Dict[str, str]]): Target variables.
            w_columns (List[Dict[str, str]]): Columns that should be treated as W.
        """
        if not set([v['name'] for v in variable_names]) <= set(self.dataset.columns) or len(variable_names) == 0:
            raise ValueError(f"The input columns (min. 1) you entered are invalid. Possible: {set(self.dataset.columns)}; input: {set([v['name'] for v in variable_names])}")

        if not set([t['name'] for t in targets]) <= set(self.dataset.columns) or len(targets) == 0:
            raise ValueError(f"The target columns (min. 1) you entered are invalid. Possible: {set(self.dataset.columns)}; input: {set([t['name'] for t in targets])}")

        if not set([w['name'] for w in w_columns]) <= set(self.dataset.columns) or len(w_columns) == 0:
            raise ValueError(f"The w columns (min. 1) you entered are invalid. Possible: {set(self.dataset.columns)}; input: {set([w['name'] for w in w_columns])}")

        def parse_column(var):
            if var['type'] == 'smiles':
                unique_smiles = self.dataset[var['name']].unique().tolist()
                return [self.smiles_to_fingerprint(smiles) for smiles in unique_smiles]
            elif var['type'] == 'scalar':
                values = self.dataset[var['name']].sort_values().unique().tolist()
                return torch.tensor([[val] for val in values])
            elif var['type'] == 'array':
                values = self.dataset[var['name']].sort_values().unique().tolist()
                return [torch.Tensor([float(i) for i in re.split(r"\s+", val)]) for val in values]
            else:
                raise ValueError(f"The variable type {var['type']} does not exist. Please choose from {{'smiles', 'scalar', 'array'}}")

        for variable in variable_names:
            self.x_options[variable['name']] = parse_column(variable)

        for w in w_columns:
            self.w_options[w['name']] = parse_column(w)
    
    def random_split(self, num_samples: int = None):
        """
        Random split of w_options into train and test.

        Args:
            num_samples (int): Number of samples to consider in the training set for each w option (same for all w).
        """

        self.w_options_train = {}
        self.w_options_test = {}

        for key, value in self.w_options.items():
            if num_samples is None or num_samples >= len(value):
                print(f"Num samples is None or bigger than the length of the list for option {key}. Same options present in the train and test set.")
                self.w_options_train[key] = value
                self.w_options_test[key] = value
            else:
                indices = list(range(len(value)))
                random.shuffle(indices)
                train_indices = indices[:num_samples]
                test_indices = indices[num_samples:]

                self.w_options_train[key] = [value[i] for i in train_indices]
                self.w_options_test[key] = [value[i] for i in test_indices]

    def single_split(self):
        """
        Split w options into test and train set. Train set contains one example (model substrate), chosen with the highest average similarity to all other substrates.
        """

        self.w_options_train = {}
        self.w_options_test = {}

        def find_fp(fingerprints):
            num_fps = len(fingerprints)
            similarity_matrix = torch.zeros(num_fps, num_fps)

            for i in range(num_fps):
                for j in range(num_fps):
                    if i != j:
                        fpi = DataStructs.ExplicitBitVect(fingerprints[i].numel())
                        fpj = DataStructs.ExplicitBitVect(fingerprints[j].numel())
                        for k in range(fingerprints[i].numel()):
                            if fingerprints[i][k] > 0:
                                fpi.SetBit(k)
                            if fingerprints[j][k] > 0:
                                fpj.SetBit(k)
                        similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(fpi, fpj)
            
            avg_similarities = similarity_matrix.mean(dim=1)
            idx = avg_similarities.argmax().item()

            return idx

        for key, value in self.w_options.items():

            idx = find_fp(value)
            fp = value[idx]

            self.w_options_train[key] = [fp]
            self.w_options_test[key] = [value[i] for i in range(len(value)) if i != idx]

    def index_split(self, idx):

        self.w_options_train = {}
        self.w_options_test = {}

        for key, value in self.w_options.items():

            self.w_options_train[key] = [value[i] for i in idx]
            self.w_options_test[key] = [val for i, val in enumerate(value) if i not in idx]

    def random_separate_split(self, num_samples):
        self.w_options_train = {}
        self.w_options_test = {}

        for i, (key, value) in enumerate(self.w_options.items()):
            if num_samples[i] is None or num_samples[i] >= len(value):
                print(f"Num samples is None or bigger than the length of the list for option {key}. Same options present in the train and test set.")
                self.w_options_train[key] = value
                self.w_options_test[key] = value
            else:
                indices = list(range(len(value)))
                random.shuffle(indices)
                train_indices = indices[:num_samples[i]]
                test_indices = indices[num_samples[i]:]

                self.w_options_train[key] = [value[j] for j in train_indices]
                self.w_options_test[key] = [value[j] for j in test_indices]

    def random_smart_split(self, num_samples):
        self.w_options_train = {}
        self.w_options_test = {}

        def find_fp(fingerprints, samples):
            num_fps = len(fingerprints)
            if num_fps <= samples: 
                print(f"Num samples is None or bigger than the length of the list. Same options present in the train and test set.")
                return list(range(num_fps))
            similarity_matrix = torch.zeros(num_fps, num_fps)

            for i in range(num_fps):
                for j in range(num_fps):
                    if i != j:
                        fpi = DataStructs.ExplicitBitVect(fingerprints[i].numel())
                        fpj = DataStructs.ExplicitBitVect(fingerprints[j].numel())
                        for k in range(fingerprints[i].numel()):
                            if fingerprints[i][k] > 0:
                                fpi.SetBit(k)
                            if fingerprints[j][k] > 0:
                                fpj.SetBit(k)
                        similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(fpi, fpj)
            
            avg_similarities = similarity_matrix.mean(dim=1)
            idx = avg_similarities.argsort(descending=True)[:samples].tolist()

            return idx
        
        for i, (key, value) in enumerate(self.w_options.items()):

            idx = find_fp(value, num_samples[i])

            self.w_options_train[key] = [value[j] for j in idx]
            self.w_options_test[key] = [val for j, val in enumerate(value) if j not in idx]

    def farthest_sample_split(self, num_samples):
        """
        Train set consists of n number of substrates - the first substrate is randomly picked.
        Subsuquent substrates are iteratively added, starting from substrates that are farthest
        from the existing train set based on Tanimoto similarity of their fingerprints.
        """
        self.w_options_train = {}
        self.w_options_test = {}

        def find_farthest_sample(fingerprints, num_train_samples):
            num_fps = len(fingerprints)
            if num_fps <= num_train_samples: 
                print(f"Num samples is bigger than the length of the list. No sampling is done.")
                return list(range(num_fps))

            similarity_matrix = torch.zeros(num_fps, num_fps)
            for i in range(num_fps): 
                for j in range(i + 1, num_fps):
                    fpi = DataStructs.ExplicitBitVect(fingerprints[i].numel())
                    fpj = DataStructs.ExplicitBitVect(fingerprints[j].numel())
                    for k in range(fingerprints[i].numel()):
                        if fingerprints[i][k] > 0:
                            fpi.SetBit(k)
                        if fingerprints[j][k] > 0:
                            fpj.SetBit(k)
                    similarity_matrix[i,j] = DataStructs.TanimotoSimilarity(fpi, fpj)
                    similarity_matrix[j, i] = DataStructs.TanimotoSimilarity(fpi, fpj)

            selected_samples = [random.randint(0, num_fps-1)]
            tanimoto_matrix = similarity_matrix[selected_samples[0]].clone()

            for _ in range(num_train_samples - 1):
                farthest_sample = tanimoto_matrix.argmin().item()
                selected_samples.append(farthest_sample) 

                tanimoto_matrix = torch.min(tanimoto_matrix,similarity_matrix[farthest_sample])

            return selected_samples           

        for i, (key,value) in enumerate(self.w_options.items()): 
            selected_indices = find_farthest_sample(value, num_train_samples=num_samples[i])

            self.w_options_train[key] = [value[j] for j in selected_indices]
            self.w_options_test[key] = [val for j, val in enumerate(value) if j not in selected_indices]
