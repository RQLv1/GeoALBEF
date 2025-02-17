import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import numpy as np
from tqdm import tqdm
from dataloader.geo_feature import atom_featurizer, bond_featurizer, Graph3dFeaturizer

def molecule_from_smiles(smiles):
    
    molecule = Chem.MolFromSmiles(smiles, sanitize= False) 

    flag = Chem.SanitizeMol(molecule, catchErrors= True) 
    if flag != Chem.SanitizeFlags.SANITIZE_NONE: 
        Chem.SanitizeMol(molecule, sanitizeOps= Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt= True, force= True)

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(molecule, params)
    AllChem.UFFOptimizeMolecule(molecule)

    AllChem.ComputeGasteigerCharges(molecule)
    molecule.SetProp('_GasteigerChargesCalculated', '1')
    
    return molecule

def graph_from_molecule(molecule):
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()]) 
            bond_features.append(bond_featurizer.encode(bond))

    g3d = Graph3dFeaturizer(molecule)
    bond_angles, bond_angle_pair_indices = g3d.get_bond_angles()
    dihedral_angles, dihedral_angle_pair_indices = g3d.get_dihedral_angles()
            
    return torch.tensor(np.array(atom_features), dtype=torch.float32), \
        torch.tensor(np.array(bond_features), dtype=torch.float32), \
        torch.tensor(np.array(pair_indices), dtype=torch.int64), \
        torch.tensor(bond_angles, dtype=torch.float32), \
        torch.tensor(dihedral_angles, dtype=torch.float32), \
        torch.tensor(bond_angle_pair_indices, dtype=torch.int64), \
        torch.tensor(dihedral_angle_pair_indices, dtype=torch.int64)

def graphs_from_smiles(smiles_list):

    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []
    bond_angles_list = []
    dihedral_angles_list = []
    bond_angle_pair_indices_list = []
    dihedral_angle_pair_indices_list = []
    smiles_list_checked = []

    for smiles in tqdm(smiles_list):
        try:
            molecule = molecule_from_smiles(smiles)
            atom_features, bond_features, pair_indices, bond_angles, dihedral_angles, bond_angle_pair_indices, dihedral_angle_pair_indices = graph_from_molecule(molecule)

            atom_features_list.append(atom_features)
            bond_features_list.append(bond_features)
            pair_indices_list.append(pair_indices)
            bond_angles_list.append(bond_angles)
            dihedral_angles_list.append(dihedral_angles)
            bond_angle_pair_indices_list.append(bond_angle_pair_indices)
            dihedral_angle_pair_indices_list.append(dihedral_angle_pair_indices)
            smiles_list_checked.append(smiles)
        except:
            continue

    return (
        atom_features_list,
        bond_features_list,
        pair_indices_list,
        bond_angles_list,
        dihedral_angles_list,
        bond_angle_pair_indices_list,
        dihedral_angle_pair_indices_list
    ), smiles_list_checked