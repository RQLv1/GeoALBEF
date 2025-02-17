import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class MPNNDataset(Dataset):
    def __init__(self, x_data, y_data) -> None:
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
    
    def __len__(self):
        return len(self.y_data[0])
    
    def __getitem__(self, idx):
        atom_features = self.x_data[0][idx]
        bond_features = self.x_data[1][idx]
        pair_indices = self.x_data[2][idx]
        bond_angles_features = self.x_data[3][idx]
        dihedral_angles_features = self.x_data[4][idx]
        bond_angle_pair_indices = self.x_data[5][idx]
        dihedral_angle_pair_indices = self.x_data[6][idx]
        smiles = self.x_data[7][idx]
        Egc = self.y_data[0][idx]
        Egb = self.y_data[1][idx]
        Eib = self.y_data[2][idx]
        Ei = self.y_data[3][idx]
        Eea = self.y_data[4][idx]
        nc = self.y_data[5][idx]
        ne = self.y_data[6][idx]
        TSb = self.y_data[7][idx]
        TSy = self.y_data[8][idx]
        YM = self.y_data[9][idx]
        permCH4 = self.y_data[10][idx]
        permCO2 = self.y_data[11][idx]
        permH2 = self.y_data[12][idx]
        permO2 = self.y_data[13][idx]
        permN2 = self.y_data[14][idx]
        permHe = self.y_data[15][idx]
        Eat = self.y_data[16][idx]
        LOI = self.y_data[17][idx]
        Xc = self.y_data[18][idx]
        Xe = self.y_data[19][idx]
        Cp = self.y_data[20][idx]
        Td = self.y_data[21][idx]
        Tg = self.y_data[22][idx]
        Tm = self.y_data[23][idx]
        return atom_features, bond_features, pair_indices, bond_angles_features, \
            dihedral_angles_features, bond_angle_pair_indices, dihedral_angle_pair_indices, smiles,\
            Egc, Egb, Eib, Ei, Eea, nc, ne, TSb, TSy, YM, permCH4, permCO2, permH2, permO2, permN2, permHe, \
            Eat, LOI, Xc, Xe, Cp, Td, Tg, Tm

def prepare_batch(batch):
    atom_features, bond_features, pair_indices, bond_angles_features, \
        dihedral_angles_features, bond_angle_pair_indices, dihedral_angle_pair_indices, smiles,\
        Egc, Egb, Eib, Ei, Eea, nc, ne, TSb, TSy, YM, permCH4, permCO2, permH2, permO2, permN2, permHe, \
        Eat, LOI, Xc, Xe, Cp, Td, Tg, Tm = zip(*batch)

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
        bond_angles_features, dihedral_angles_features, bond_angle_pair_indices, dihedral_angle_pair_indices, smiles), \
        (Egc, Egb, Eib, Ei, Eea, nc, ne, TSb, TSy, YM, permCH4, permCO2, permH2, permO2, permN2, permHe, Eat, 
        LOI, Xc, Xe, Cp, Td, Tg, Tm)