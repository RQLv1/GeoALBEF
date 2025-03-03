import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from dataloader.geo_utils import RBF


class Featurizer:
    def __init__(self, allowable_sets, continuous_features= None) -> None:
        self.dim = 0 
        self.features_mapping = {}
        self.continuous_features = continuous_features or set()

        for k, s in allowable_sets.items():
            if k in self.continuous_features:
                if k == "partial_charge":
                    feature_dim = 10 
                    self.features_mapping[k] = slice(self.dim, self.dim + feature_dim)
                    self.dim += feature_dim
                elif k == "mass": 
                    feature_dim = 10 
                    self.features_mapping[k] = slice(self.dim, self.dim + feature_dim)
                    self.dim += feature_dim
                elif k == "pos": 
                    feature_dim = 3 
                    self.features_mapping[k] = slice(self.dim, self.dim + feature_dim)
                    self.dim += feature_dim
                elif k == "length":
                    feature_dim = 10 
                    self.features_mapping[k] = slice(self.dim, self.dim + feature_dim)
                    self.dim += feature_dim
                else:
                    self.features_mapping[k] = slice(self.dim, self.dim + 1)
                    self.dim += 1
            else:
                s = sorted(list(s))
                self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
                self.dim += len(s)
    
    def encode(self, inputs):
        output = np.zeros((self.dim))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if name_feature in self.continuous_features:
                if isinstance(feature, np.ndarray):
                    feature_size = len(feature.flatten())
                    start = feature_mapping.start
                    output[start:start+feature_size] = feature.flatten()
                else:
                    output[feature_mapping] = feature
            else:
                output[feature_mapping[feature]] = 1
        return output

class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets, continuous_features= None) -> None:
        super().__init__(allowable_sets, continuous_features)
        
    #discrete
    def symbol(self, atom):
        if atom.GetSymbol() == '*' or atom.GetSymbol() == 'R':
            return '*'
        return atom.GetSymbol()
    def n_valence(self, atom):
        if atom.GetSymbol() == '*' or atom.GetSymbol() == 'R':
            return 0
        return atom.GetTotalValence()
    def n_hydrogens(self, atom):
        if atom.GetSymbol() == '*' or atom.GetSymbol() == 'R':
            return 0
        return atom.GetTotalNumHs()
    def hybridization(self, atom):
        if atom.GetSymbol() == '*' or atom.GetSymbol() == 'R':
            return 's'
        return atom.GetHybridization().name.lower()
    
    #continuous
    def mass(self, atom):
        atmass = atom.GetMass()
        atmass_list = RBF(np.arange(0, 20, 2), 3, np.array([[atmass]]))
        atmass_list[atmass_list < 1e-8] = 0
        return atmass_list

    def partial_charge(self, atom):
        mol = atom.GetOwningMol()
        partial_charge = float(atom.GetProp('_GasteigerCharge'))
        partial_charge_list = None
        if np.isnan(partial_charge):
            partial_charge_list = np.zeros(10)
        else:
            partial_charge_list = RBF(np.arange(-1, 4, 0.5), 3, np.array([[partial_charge]]))
            partial_charge_list[partial_charge_list < 1e-8] = 0
        return partial_charge_list
    
    def pos(self, atom): # 3d position
        conformer = atom.GetOwningMol().GetConformer()
        pos = conformer.GetAtomPosition(atom.GetIdx())
        return np.array([pos.x, pos.y, pos.z])

class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets, continuous_features= None) -> None:
        super().__init__(allowable_sets, continuous_features)
        self.dim += 1 
    def encode(self, bond):
        output = np.zeros((self.dim))
        if bond is None:
            output[-1] = 1
            return output
        output = super().encode(bond)
        return output

    #discrete
    def bond_type(self, bond):
        return bond.GetBondType().name.lower()
    def conjugated(self, bond):
        return bond.GetIsConjugated()

    #continuous
    def length(self, bond):
        conformer = bond.GetOwningMol().GetConformer()
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()

        atom1_pos = conformer.GetAtomPosition(atom1_idx)
        atom2_pos = conformer.GetAtomPosition(atom2_idx)

        distance = np.linalg.norm(np.array(atom1_pos) - np.array(atom2_pos))

        distance_list = RBF(np.arange(0, 3, 0.3), 3, np.array([[distance]]))
        distance_list[distance_list < 1e-8] = 0
        return distance_list

class Graph3dFeaturizer():
    def __init__(self, molecule):
        self.molecule = molecule
        self.conformer = molecule.GetConformer()
    
    def get_bond_angles(self):
        
        angles = []
        positions = []
        angle_indices = []
        bond_angle_pair_indices = []
        for atom in self.molecule.GetAtoms():
            pos = self.conformer.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
        positions = np.array(positions)
        for bond in self.molecule.GetBonds():
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()
            
            for central_idx in [atom1_idx, atom2_idx]:
                other_idx = atom2_idx if central_idx == atom1_idx else atom1_idx
                central_atom = self.molecule.GetAtomWithIdx(central_idx)
                
                for neighbor in central_atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx != other_idx:
                        v1 = positions[other_idx] - positions[central_idx]
                        v2 = positions[neighbor_idx] - positions[central_idx]
                        
                        v1_norm = v1 / np.linalg.norm(v1)
                        v2_norm = v2 / np.linalg.norm(v2)
                        
                        cos_angle = np.dot(v1_norm, v2_norm)
                        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                        angle_atoms = sorted([other_idx, central_idx, neighbor_idx])
                        if angle_atoms not in angle_indices:
                            angles.extend([angle]*  2)
                            angle_indices.append(angle_atoms)
                            bond_angle_pair_indices.append([central_idx, neighbor_idx])
                            bond_angle_pair_indices.append([central_idx, other_idx])
        angles = np.array(angles)
        bond_angle_pair_indices = np.array(bond_angle_pair_indices)
        return RBF(np.arange(0, 3, 0.3), 3, angles), bond_angle_pair_indices

    def get_dihedral_angles(self):
        dihedral_angles = []
        dihedral_angle_pair_indices = []
        sequence_exclude = self._get_sequence()
        for seq in sequence_exclude:
            atom_idx1, atom_idx2, atom_idx3, atom_idx4 = seq[0], seq[1], seq[2], seq[3]

            atom1 = self.molecule.GetAtomWithIdx(atom_idx1)
            atom2 = self.molecule.GetAtomWithIdx(atom_idx2)
            atom3 = self.molecule.GetAtomWithIdx(atom_idx3)
            atom4 = self.molecule.GetAtomWithIdx(atom_idx4)

            pos1 = self.conformer.GetAtomPosition(atom_idx1)
            pos2 = self.conformer.GetAtomPosition(atom_idx2)
            pos3 = self.conformer.GetAtomPosition(atom_idx3)
            pos4 = self.conformer.GetAtomPosition(atom_idx4)

            pos1 = np.array([pos1.x, pos1.y, pos1.z])
            pos2 = np.array([pos2.x, pos2.y, pos2.z])
            pos3 = np.array([pos3.x, pos3.y, pos3.z])
            pos4 = np.array([pos4.x, pos4.y, pos4.z])

            pos12 = pos2 - pos1
            pos23 = pos3 - pos2
            pos34 = pos4 - pos3

            n1 = np.cross(pos12, pos23)
            n2 = np.cross(pos23, pos34)

            cos_phi = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
            cos_phi = np.clip(cos_phi, -1.0, 1.0)
            phi_rad = np.arccos(cos_phi)
            dihedral_angles.extend([phi_rad]*4)
            dihedral_angle_pair_indices.append([atom_idx1, atom_idx2])
            dihedral_angle_pair_indices.append([atom_idx2, atom_idx3])
            dihedral_angle_pair_indices.append([atom_idx3, atom_idx4])
            dihedral_angle_pair_indices.append([atom_idx4, atom_idx1])

        dihedral_angles = np.array(dihedral_angles)
        dihedral_angle_pair_indices = np.array(dihedral_angle_pair_indices)
        
        return RBF(np.arange(0, 3, 0.3), 3, dihedral_angles), dihedral_angle_pair_indices

    def _get_distance(self, atom):
        neighbor_distances = []
        for neighbor in atom.GetNeighbors():
            bond = self.molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            conformer = bond.GetOwningMol().GetConformer()
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()
            atom1_pos = conformer.GetAtomPosition(atom1_idx)
            atom2_pos = conformer.GetAtomPosition(atom2_idx)
            neighbor_distances.append(np.linalg.norm(np.array(atom1_pos) - np.array(atom2_pos)))
        return neighbor_distances

    def _find_atom_sequence(self, atom_a0):
        
        atom_a1 = self._get_next_closet_neighbor(atom_a0, {atom_a0.GetIdx()})
        if not atom_a1:
            return None

        excluded = {atom_a0.GetIdx(), atom_a1.GetIdx()}
        atom_a2 = self._get_next_closet_neighbor(atom_a1, excluded) or \
            self._get_next_closet_neighbor(atom_a0, excluded)
        if not atom_a2:
            return None

        excluded.add(atom_a2.GetIdx())
        atom_a3 = self._get_next_closet_neighbor(atom_a2, excluded) or \
            self._get_next_closet_neighbor(atom_a1, excluded) or \
            self._get_next_closet_neighbor(atom_a0, excluded)
        if not atom_a3:
            return None
        
        return atom_a0, atom_a1, atom_a2, atom_a3

    def _get_next_closet_neighbor(self, center_atom, exclude_atoms):
            distances = self._get_distance(center_atom)
            for idx in np.argsort(distances):
                neighbor = center_atom.GetNeighbors()[idx]
                if neighbor.GetIdx() not in exclude_atoms:
                    return neighbor
            return None

    def _get_sequence(self):
        sequence = []
        for atom in self.molecule.GetAtoms():
            result = self._find_atom_sequence(atom)
            if result is not None:
                a, b, c, d = result
            sequence.append([a.GetIdx(), b.GetIdx(), c.GetIdx(), d.GetIdx()])
    
        sequence_exclude = []
        seen = set()
        for seq in sequence:
            seq_set = frozenset(seq)
            if seq_set not in seen:
                sequence_exclude.append(seq)
                seen.add(seq_set)
        return sequence_exclude
        
        

atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 
                    'Cl', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Cs', 'Ni', 'Cu', 'Zn', 
                     'Br', 'Y', 'Zr', 'Nb', 'Mo', 'Rh', 'Pd', 'Ag', 'Sn', 'Sb', 'I', 'Ba',  'Au', 'Hg', '*'},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3", "sp3d", "sp3d2"},
        "mass": None,
        "partial_charge": None,
        "pos": None,
    },
    continuous_features= {"mass", "partial_charge", "pos"}
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
        "length": None,
    },
    continuous_features= {"length"}
)