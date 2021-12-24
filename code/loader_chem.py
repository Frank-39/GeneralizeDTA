import networkx as nx
import numpy as np
import json
import os
import torch
import pickle
from torch_geometric.utils import add_remaining_self_loops,add_self_loops
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from itertools import repeat, product, chain
from torch_geometric import data as DATA

from plus.model import plus_tfm,transformer
def mol_to_graph_data_obj_simple2(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data
def preprocess_seq(x0, num_alphabets=21, max_len=1000):
    special_tokens = {"MASK": torch.tensor([num_alphabets], dtype=torch.long),
                      "CLS": torch.tensor([num_alphabets + 1], dtype=torch.long),
                      "SEP": torch.tensor([num_alphabets + 2], dtype=torch.long)}
    tokens = torch.zeros(max_len, dtype=torch.long)
    segments = torch.zeros(max_len, dtype=torch.long)
    input_mask = torch.zeros(max_len, dtype=torch.bool)
    max_len -= 2
    x0 = x0[:max_len]

    tokens[:len(x0) + 2] = torch.cat([special_tokens["CLS"], x0, special_tokens["SEP"]])
    input_mask[:len(x0) + 2] = True
    return tokens, segments, input_mask
def preprocess_seq2(x0, num_alphabets=21, max_len=1024):
    special_tokens = {"MASK": torch.tensor([num_alphabets], dtype=torch.long),
                      "CLS": torch.tensor([num_alphabets + 1], dtype=torch.long),
                      "SEP": torch.tensor([num_alphabets + 2], dtype=torch.long)}
    tokens = [0,0]
    segments=[0,0]
    input_mask = [0,0]
    tokens[0] = torch.zeros(max_len//2, dtype=torch.long)
    segments[0] = torch.zeros(max_len//2, dtype=torch.long)
    input_mask[0] = torch.zeros(max_len//2, dtype=torch.bool)
    max_len -= 2
    x1 = x0[:max_len//2]
    tokens[0][:len(x0) + 2] = torch.cat([special_tokens["CLS"], x1, special_tokens["SEP"]])
    input_mask[0][:len(x0) + 2] = True

    tokens[1] = torch.zeros(max_len// 2, dtype=torch.long)
    segments[1] = torch.zeros(max_len // 2, dtype=torch.long)
    input_mask[1] = torch.zeros(max_len // 2, dtype=torch.bool)
    max_len -= 2
    x2 = x0[max_len//2:max_len]
    tokens[1][:(len(x0) + 2)] = torch.cat([special_tokens["CLS"], x2, special_tokens["SEP"]])
    input_mask[1][:len(x0) + 2] = True


    return tokens, segments, input_mask
# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.float32)
    else:  # mol has no bonds

        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return x,edge_index,edge_attr



class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 # data = None,
                 # slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []

        if self.dataset == 'zinc_standard_agent':
            print('我在执行这个1')
            input_path = self.raw_paths[0]
            input_df = pd.read_csv(input_path, sep=',', compression='gzip',
                                   dtype='str')
            smiles_list = list(input_df['smiles'])
            zinc_id_list = list(input_df['zinc_id'])
            # for i in range(len(smiles_list)):
            print(len(smiles_list))
            for i in range(100000):

                s = smiles_list[i]
                # each example contains a single species
                try:
                    rdkit_mol = AllChem.MolFromSmiles(s)
                    if rdkit_mol != None:  # ignore invalid mol objects
                        # # convert aromatic bonds to double bonds
                        # Chem.SanitizeMol(rdkit_mol,
                        #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                        data = mol_to_graph_data_obj_simple2(rdkit_mol)
                        # manually add mol id
                        ids= int(zinc_id_list[i].split('ZINC')[1].lstrip('0'))
                        data.id = torch.tensor(
                            [ids])  # id here is zinc id value, stripped of
                        # leading zeros
                        data_list.append(data)
                        data_smiles_list.append(smiles_list[i])
                except:
                    continue
        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# 蛋白质的操作


class Alphabet:
    """ biological sequence encoder """
    def __init__(self, chars, encoding, chars_rc=None, encoding_rc=None, missing=255):
        self.chars = np.frombuffer(chars, dtype=np.uint8)
        self.encoding = np.zeros(256, dtype=np.uint8) + missing
        self.encoding[self.chars] = encoding
        if chars_rc is not None:
            self.chars_rc = np.frombuffer(chars_rc, dtype=np.uint8)
            self.encoding_rc = np.zeros(256, dtype=np.uint8) + missing
            self.encoding_rc[self.chars_rc] = encoding_rc
        self.size = encoding.max() + 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return chr(self.chars[i])

    def encode(self, x, reverse_complement=False):
        """ encode a byte string into alphabet indices """
        if not reverse_complement:
            x = np.frombuffer(x, dtype=np.uint8)
            string = self.encoding[x]
        else:
            x = np.frombuffer(x, dtype=np.uint8)[::-1]
            string = self.encoding_rc[x]
        return string

    def decode(self, x, reverse_complement=False):
        """ decode index array, x, to byte string of this alphabet """
        if not reverse_complement:
            string = self.chars[x-1]
        else:
            string = self.chars_rc[x[::-1]-1]
        return string.tobytes()
class Protein(Alphabet):
    """ protein sequence encoder """
    def __init__(self):
        chars = b'ARNDCQEGHILKMFPSTWYVXOUBZ'
        encoding = np.arange(len(chars))
        encoding[21:] = [11, 4, 20, 20]  # encode 'OUBZ' as synonyms
        encoding += 1                    # leave 0 for padding tokens
        super(Protein, self).__init__(chars, encoding, missing=21)
def preprocess_seq(x0, num_alphabets=21, max_len=1000):
    special_tokens = {"MASK": torch.tensor([num_alphabets], dtype=torch.long),
                      "CLS": torch.tensor([num_alphabets + 1], dtype=torch.long),
                      "SEP": torch.tensor([num_alphabets + 2], dtype=torch.long)}
    tokens = torch.zeros(max_len, dtype=torch.long)
    segments = torch.zeros(max_len, dtype=torch.long)
    input_mask = torch.zeros(max_len, dtype=torch.bool)
    max_len -= 2
    x0 = x0[:max_len]

    tokens[:len(x0) + 2] = torch.cat([special_tokens["CLS"], x0, special_tokens["SEP"]])
    input_mask[:len(x0) + 2] = True
    return tokens, segments, input_mask
class Cp_dataset(InMemoryDataset):
    def __init__(self,
                 root,
                 # data = None,
                 # slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='cp_dataset',
                 empty=False):
        self.dataset = dataset
        self.root = root

        super(Cp_dataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)

        if not empty :
            if self.dataset == "train":
                self.data, self.slices = torch.load('../data/davis/processed/geometric_datatrain_processed.pt')
            elif self.dataset == "test":
                self.data, self.slices = torch.load("../data/davis/processed/geometric_datatest_processed.pt")
            elif self.dataset == "dev":
                self.data, self.slices = torch.load("../data/davis/processed/geometric_datadev_processed.pt")


    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data'+str(self.dataset)+'_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')
    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data.to('cuda')
    def process(self):

        data_list = []
        alphabet=Protein()
        # self.dataset == 'train':
        print("我在执行cp_",self.dataset,"_dataset")
        input_path = self.root + '/raw/davis_'+str(self.dataset)+'.csv'
        input_df = pd.read_csv(input_path)
        compound_iso_smiles = np.asarray(list(input_df['compound_iso_smiles']))
        target_sequence = np.asarray(list(input_df['target_sequence']))
        affinity = np.asarray(list(input_df['affinity']))


        sequences = []
        for s in target_sequence:
            s = s.encode('utf-8')
            sequence = alphabet.encode(s.upper())
            sequences.append(sequence)
        for i in range(len(compound_iso_smiles)):
        # for i in range(2000):
            if i % 100==0:
                print(i)
            s = compound_iso_smiles[i]

            rdkit_mol = AllChem.MolFromSmiles(s)
            if rdkit_mol != None:  # ignore invalid mol objects
                x,edge_index,edge_attr = mol_to_graph_data_obj_simple(rdkit_mol)

                y = affinity[i]

                target = torch.tensor(sequences[i],dtype=torch.long)

                # target_token,segement,input_mask = preprocess_seq(torch.from_numpy(target),21,1000)
                target_token,segement,input_mask = preprocess_seq(target,21,512)

                # print(x,y,target)
                GCNData = DATA.Data(x=x,
                                    edge_index=edge_index,
                                    edge_attr=edge_attr,
                                    y=torch.FloatTensor([y]))
                GCNData.target_token = target_token
                GCNData.segments = segement
                GCNData.input_mask = input_mask
                data_list.append(GCNData)
            else:
                print(i)

        data, slices = self.collate(data_list)
        print(len(data_list))
        # if self.dataset =='train':
        torch.save((data, slices), self.processed_paths[0])
        # else:
        #     torch.save((data, slices), self.processed_paths[1])

if __name__ == "__main__":
    pass