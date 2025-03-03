import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeNetwork(nn.Module):
    def __init__(self, atom_dim, bond_dim) -> None:
        super().__init__()
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.kernel = nn.Parameter(torch.empty(self.bond_dim, self.atom_dim * self.atom_dim))
        self.bias = nn.Parameter(torch.zeros(self.atom_dim * self.atom_dim))
        nn.init.xavier_uniform_(self.kernel)
    
    def forward(self, atom_features, bond_features, pair_indices):
        
        bond_features = torch.matmul(bond_features, self.kernel) + self.bias 
        bond_features = bond_features.view(-1, self.atom_dim, self.atom_dim) 

        atom_features_neighbors = torch.gather(atom_features, 0, pair_indices[:, 1].unsqueeze(-1).expand(-1, atom_features.size(-1)))
        atom_features_neighbors = atom_features_neighbors.unsqueeze(-1) 

        transformed_features = torch.matmul(bond_features, atom_features_neighbors).squeeze(-1)
        aggregated_bond_features = torch.zeros_like(atom_features)
        aggregated_bond_features.scatter_add_(0, pair_indices[:, 0].unsqueeze(-1).expand(-1, transformed_features.size(1)), transformed_features)

        return aggregated_bond_features

class BondAngleNetwork(nn.Module):
    def __init__(self, atom_dim, bond_angle_dim) -> None:
        super().__init__()
        self.atom_dim = atom_dim
        self.bond_angle_dim = bond_angle_dim
        self.kernel = nn.Parameter(torch.empty(self.bond_angle_dim, self.atom_dim * self.atom_dim))
        self.bias = nn.Parameter(torch.zeros(self.atom_dim * self.atom_dim))
        nn.init.xavier_uniform_(self.kernel)
    
    def forward(self, atom_features, bond_angle_features, bond_angle_pair_indices):
         bond_angle_features = torch.matmul(bond_angle_features, self.kernel) + self.bias
         bond_angle_features = bond_angle_features.view(-1, self.atom_dim, self.atom_dim)

         atom_features_neighbors = torch.gather(atom_features, 0, bond_angle_pair_indices[:, 1].unsqueeze(-1).expand(-1, atom_features.size(-1)))
         atom_features_neighbors = atom_features_neighbors.unsqueeze(-1)

         transformed_features = torch.matmul(bond_angle_features, atom_features_neighbors).squeeze(-1)
         aggregated_bond_angle_features = torch.zeros_like(atom_features)
         aggregated_bond_angle_features.scatter_add_(0, bond_angle_pair_indices[:, 0].unsqueeze(-1).expand(-1, transformed_features.size(1)), transformed_features)

         return aggregated_bond_angle_features

class DihedralAngleNetwork(nn.Module):
    def __init__(self, atom_dim, dihedral_angle_dim) -> None:
        super().__init__()
        self.atom_dim = atom_dim
        self.dihedral_angle_dim = dihedral_angle_dim
        self.kernel = nn.Parameter(torch.empty(self.dihedral_angle_dim, self.atom_dim * self.atom_dim))
        self.bias = nn.Parameter(torch.zeros(self.atom_dim * self.atom_dim))
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, atom_features, dihedral_angle_features, dihedral_angle_pair_indices):
        dihedral_angle_features = torch.matmul(dihedral_angle_features, self.kernel) + self.bias
        dihedral_angle_features = dihedral_angle_features.view(-1, self.atom_dim, self.atom_dim)

        atom_features_neighbors = torch.gather(atom_features, 0, dihedral_angle_pair_indices[:, 1].unsqueeze(-1).expand(-1, atom_features.size(-1)))
        atom_features_neighbors = atom_features_neighbors.unsqueeze(-1)

        transformed_features = torch.matmul(dihedral_angle_features, atom_features_neighbors).squeeze(-1)
        aggregated_dihedral_angle_features = torch.zeros_like(atom_features)
        aggregated_dihedral_angle_features.scatter_add_(0, dihedral_angle_pair_indices[:, 0].unsqueeze(-1).expand(-1, transformed_features.size(1)), transformed_features)

        return aggregated_dihedral_angle_features

class MessagePassing(nn.Module):
    def __init__(self, atom_dim, bond_dim, bond_angle_dim, dihedral_angle_dim, units, steps=4) -> None:
        super().__init__()
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.bond_angle_dim = bond_angle_dim
        self.dihedral_angle_dim = dihedral_angle_dim

        self.units = units
        self.steps = steps
        self.pad_length = max(0, self.units - self.atom_dim)

        self.message_step_edge = EdgeNetwork(self.atom_dim + self.pad_length, self.bond_dim)
        self.message_step_bond_angle = BondAngleNetwork(self.atom_dim + self.pad_length, self.bond_angle_dim)
        self.message_step_dihedral_angle = DihedralAngleNetwork(self.atom_dim + self.pad_length, self.dihedral_angle_dim)
        self.update_step = nn.GRUCell(self.atom_dim + self.pad_length, self.atom_dim + self.pad_length, dtype= torch.float32)
        
    def forward(self, 
                atom_features, 
                bond_features, 
                pair_indices, 
                bond_angle_features,
                bond_angle_pair_indices, 
                dihedral_angle_features, 
                dihedral_angle_pair_indices):

        if self.pad_length > 0:
            atom_features_updated = F.pad(atom_features, (0, self.pad_length))
        else:
            atom_features_updated = atom_features

        for i in range(self.steps):
            atom_features_aggregated = self.message_step_edge(
                atom_features_updated, bond_features, pair_indices
            )
            atom_features_aggregated_bond_angle = self.message_step_bond_angle(
                atom_features_aggregated, bond_angle_features, bond_angle_pair_indices
            )
            atom_features_aggregated_dihedral_angle = self.message_step_dihedral_angle(
                atom_features_aggregated_bond_angle, dihedral_angle_features, dihedral_angle_pair_indices
            )

            atom_features_updated = self.update_step(
                atom_features_aggregated_dihedral_angle, atom_features_updated
            )

        return atom_features_updated

class PartitionPadding(nn.Module):
    def __init__(self, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size
    
    def forward(self, inputs):
        atom_features, molecule_indicator = inputs

        atom_features_partitioned = [atom_features[molecule_indicator == i] for i in range(self.batch_size)]

        num_atoms = [f.size(0) for f in atom_features_partitioned]
        max_num_atoms = max(num_atoms)

        atom_features_stacked = torch.stack(
            [
                F.pad(f, (0, 0, 0, max_num_atoms - n))
                for f,n in zip(atom_features_partitioned, num_atoms)
            ]
        )

        gather_indices = torch.nonzero(atom_features_stacked.sum(dim= (1, 2)) != 0).squeeze()

        return atom_features_stacked[gather_indices]

class TransformerEncoderReadout(nn.Module):

    def __init__(self, num_heads= 8, embed_dim= 64, dense_dim= 512 ,batch_size= 64) -> None:
        super().__init__()

        self.partition_padding = PartitionPadding(batch_size)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dtype= torch.float32)
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, dense_dim, dtype= torch.float32),
            nn.ReLU(),
            nn.Linear(dense_dim, dense_dim, dtype= torch.float32),
        )
        # self.average_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, inputs):
        x = self.partition_padding(inputs)

        padding_mask = (x.sum(dim= -1) != 0)

        attention_output, _ = self.attention(x, x, x, key_padding_mask= padding_mask.float().transpose(0, 1))

        proj_output = self.dense_proj(attention_output)

        return  proj_output #self.average_pooling(proj_output.permute(0, 2, 1)).squeeze(-1)


class MPNNModel(nn.Module):
    def __init__(self, 
                atom_dim, 
                bond_dim, 
                bond_angle_dim, 
                dihedral_angle_dim, 
                batch_size= 64, 
                message_units= 64, 
                message_steps=5, 
                num_attention_heads= 8, 
                dense_units= 512) -> None:
        super().__init__()

        self.message_passing = MessagePassing(atom_dim, 
                                            bond_dim, 
                                            bond_angle_dim, 
                                            dihedral_angle_dim, 
                                            message_units, 
                                            message_steps)

        self.transformer_encoder = TransformerEncoderReadout(num_heads=num_attention_heads, 
                                                             embed_dim=message_units, 
                                                             dense_dim=dense_units, 
                                                             batch_size=batch_size)

    def forward(self, 
               atom_features, 
               bond_features, 
               pair_indices, 
               molecule_indicator, 
               bond_angle_features, 
               bond_angle_pair_indices,
               dihedral_angle_features, 
               dihedral_angle_pair_indices):

        x = self.message_passing(atom_features = atom_features, 
                                 bond_features = bond_features, 
                                 pair_indices = pair_indices, 
                                 bond_angle_features = bond_angle_features, 
                                 bond_angle_pair_indices = bond_angle_pair_indices, 
                                 dihedral_angle_features = dihedral_angle_features, 
                                 dihedral_angle_pair_indices = dihedral_angle_pair_indices)

        x = self.transformer_encoder([x, molecule_indicator])
        return x
