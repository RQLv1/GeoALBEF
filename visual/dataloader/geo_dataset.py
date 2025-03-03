import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class MPNNDataset(Dataset):
    def __init__(self, x_data, y_data) -> None:
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        atom_features = self.x_data[0][idx]
        bond_features = self.x_data[1][idx]
        pair_indices = self.x_data[2][idx]
        bond_angles_features = self.x_data[3][idx]
        dihedral_angles_features = self.x_data[4][idx]
        bond_angle_pair_indices = self.x_data[5][idx]
        dihedral_angle_pair_indices = self.x_data[6][idx]
        smiles = self.x_data[7][idx]
        return atom_features, bond_features, pair_indices, bond_angles_features, \
            dihedral_angles_features, bond_angle_pair_indices, dihedral_angle_pair_indices, smiles

def prepare_batch(batch):
    atom_features, bond_features, pair_indices, bond_angles_features, \
        dihedral_angles_features, bond_angle_pair_indices, dihedral_angle_pair_indices, smiles = zip(*batch)

    atoms_num_list = torch.tensor([atom_features[i].size(0) for i in range(len(atom_features))]) 
    bonds_num_list = torch.tensor([bond_features[i].size(0) for i in range(len(bond_features))]) 
    bond_angles_num_list = torch.tensor([bond_angles_features[i].size(0) for i in range(len(bond_angles_features))])
    dihedral_angles_num_list = torch.tensor([dihedral_angles_features[i].size(0) for i in range(len(dihedral_angles_features))])
    
    molecule_indices = torch.arange(len(atoms_num_list))
    molecule_indicator = torch.repeat_interleave(molecule_indices, atoms_num_list)
    
    #bond
    gather_indices = torch.repeat_interleave(molecule_indices[:-1], bonds_num_list[1:])
    increment = torch.cumsum(atoms_num_list[:-1], dim=0)
    increment = increment[gather_indices]
    increment = F.pad(increment, (bonds_num_list[0], 0)) 
    pair_indices = torch.cat(pair_indices, dim= 0)
    pair_indices = pair_indices.view(-1, pair_indices.size(-1)) + increment[:, None]
    
    #bond_angle
    bond_angle_gather_indices = torch.repeat_interleave(molecule_indices[:-1], bond_angles_num_list[1:])
    bond_angle_increment = torch.cumsum(atoms_num_list[:-1], dim=0)
    bond_angle_increment = bond_angle_increment[bond_angle_gather_indices]
    bond_angle_increment = F.pad(bond_angle_increment, (bond_angles_num_list[0], 0))
    bond_angle_pair_indices = torch.cat(bond_angle_pair_indices, dim=0)
    bond_angle_pair_indices = bond_angle_pair_indices.view(-1, bond_angle_pair_indices.size(-1)) + bond_angle_increment[:, None]
    
    #dihedral_angle
    dihedral_angle_gather_indices = torch.repeat_interleave(molecule_indices[:-1], dihedral_angles_num_list[1:])
    dihedral_angle_increment = torch.cumsum(atoms_num_list[:-1], dim=0)
    dihedral_angle_increment = dihedral_angle_increment[dihedral_angle_gather_indices]
    dihedral_angle_increment = F.pad(dihedral_angle_increment, (dihedral_angles_num_list[0], 0))
    dihedral_angle_pair_indices = torch.cat(dihedral_angle_pair_indices, dim=0)
    dihedral_angle_pair_indices = dihedral_angle_pair_indices.view(-1, dihedral_angle_pair_indices.size(-1)) + dihedral_angle_increment[:, None]
    
    atom_features = torch.cat([atoms for atoms in atom_features], dim=0)
    bond_features = torch.cat([bonds for bonds in bond_features], dim=0)
    bond_angles_features = torch.cat([bond_angles for bond_angles in bond_angles_features], dim=0)
    dihedral_angles_features = torch.cat([dihedral_angles for dihedral_angles in dihedral_angles_features], dim=0)

    return (atom_features, bond_features, pair_indices, molecule_indicator, \
        bond_angles_features, dihedral_angles_features, bond_angle_pair_indices, dihedral_angle_pair_indices, smiles)