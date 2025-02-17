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

def graphs_from_smiles(polymer_data):

    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []
    bond_angles_list = []
    dihedral_angles_list = []
    bond_angle_pair_indices_list = []
    dihedral_angle_pair_indices_list = []
    smiles_list_checked = []
    Egc_list = []
    Egb_list = []
    Eib_list = []
    Ei_list = []
    Eea_list = []
    nc_list = []
    ne_list = []
    TSb_list = []
    TSy_list = []
    YM_list = []
    permCH4_list = []
    permCO2_list = []
    permH2_list = []
    permO2_list = []
    permN2_list = []
    permHe_list = []
    Eat_list = []
    LOI_list = []
    Xc_list = []
    Xe_list = []
    Cp_list = []
    Td_list = []
    Tg_list = []
    Tm_list = []

    for index, row in tqdm(polymer_data.iterrows()):
        smiles = row['smiles']
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
            Egc_list.append(row['Egc'])
            Egb_list.append(row['Egb'])
            Eib_list.append(row['Eib'])
            Ei_list.append(row['Ei'])
            Eea_list.append(row['Eea'])
            nc_list.append(row['nc'])
            ne_list.append(row['ne'])
            TSb_list.append(row['TSb'])
            TSy_list.append(row['TSy'])
            YM_list.append(row['YM'])
            permCH4_list.append(row['permCH4'])
            permCO2_list.append(row['permCO2'])
            permH2_list.append(row['permH2'])
            permO2_list.append(row['permO2'])
            permN2_list.append(row['permN2'])
            permHe_list.append(row['permHe'])
            Eat_list.append(row['Eat'])
            LOI_list.append(row['LOI'])
            Xc_list.append(row['Xc'])
            Xe_list.append(row['Xe'])
            Cp_list.append(row['Cp'])
            Td_list.append(row['Td'])
            Tg_list.append(row['Tg'])
            Tm_list.append(row['Tm'])
        except:
            continue

    return (atom_features_list,
            bond_features_list,
            pair_indices_list,
            bond_angles_list,
            dihedral_angles_list,
            bond_angle_pair_indices_list,
            dihedral_angle_pair_indices_list,
            smiles_list_checked), \
            (Egc_list, Egb_list, Eib_list, Ei_list, Eea_list, 
             nc_list, ne_list, 
             TSb_list, TSy_list, YM_list, 
             permCH4_list, permCO2_list, permH2_list, permO2_list, permN2_list, permHe_list, 
             Eat_list, LOI_list, Xc_list, Xe_list, Cp_list, 
             Td_list, Tg_list, Tm_list)