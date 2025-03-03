import warnings
warnings.filterwarnings('ignore')
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import os

from dataloader.geo_data_utils import graphs_from_smiles, molecule_from_smiles
from dataloader.geo_dataset import MPNNDataset, prepare_batch
from models.model_pretrain import ALBEF
from models.tokenization_bert import BertTokenizer
from models.xbert import BertForMaskedLM, BertConfig
from utils import create_logger, save_checkpoint, load_checkpoint

class FunctionalGroup:
    def __init__(self):
        super().__init__()
        import json
        with open('functional_groups_eng.json', 'r') as f:
            functional_dict = json.load(f)
        self.functional_patterns = functional_dict

    def get_functional_group(self, mol, atom_idx, radius=1):
        atoms = set()
        atoms.add(atom_idx)
        start_atom = mol.GetAtomWithIdx(atom_idx)
    
        for i in range(radius):
            new_atoms = set()
            for atom_id in atoms:
                atom = mol.GetAtomWithIdx(atom_id)

                for smarts_pattern in self.functional_patterns.keys():
                    pattern = Chem.MolFromSmarts(smarts_pattern)
                    if pattern:
                        matches = mol.GetSubstructMatches(pattern)
                        for match in matches:
                            if atom_id in match:
                                new_atoms.update(match)
    
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx not in new_atoms:
                        new_atoms.add(neighbor_idx)
                        if neighbor.GetSymbol() in ['O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']:
                            for next_neighbor in neighbor.GetNeighbors():
                                next_neighbor_idx = next_neighbor.GetIdx()
                                if next_neighbor_idx not in new_atoms:
                                    new_atoms.add(next_neighbor_idx)
    
            atoms.update(new_atoms)
    
        return list(atoms)
    
    def identify_functional_group(self, mol, atom_indices):
        submol = Chem.PathToSubmol(mol, list(atom_indices))
    
        max_match_size = 0
        best_match = None
        
        for smarts, name in self.functional_patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    if any(idx in atom_indices for idx in match):
                        if len(match) > max_match_size:
                            max_match_size = len(match)
                            best_match = name
                            group_atoms = match
        
        if not best_match:
            return "unkonw", list(atom_indices)
        return best_match, list(group_atoms)
    
def get_functional_group_attn_count(smiles, weights):
    fg = FunctionalGroup()
    assert len(smiles) > 0
    assert weights is not None

    mol = molecule_from_smiles(smiles)

    actual_atoms = mol.GetNumAtoms()
    first_mol_attention = torch.diagonal(weights, dim1=1, dim2=2).transpose(0, 1).squeeze().cpu().detach().numpy()
    mask = first_mol_attention != 0
    row_means = np.sum(first_mol_attention, axis=1) / np.count_nonzero(mask, axis=1)
    threshold = 0
    mask = row_means >= threshold 
    indices = np.where(mask)[0]
    target_atoms = [int(idx) for idx in indices]
    highlight_atoms = []
    used_atoms = set() 
    target_atoms_sorted = sorted(target_atoms, 
                               key=lambda x: np.sum(row_means[x]),
                               reverse=True)
    for idx in target_atoms_sorted:
    
        if idx in used_atoms:
            continue
        
        current_group = fg.get_functional_group(mol=mol, atom_idx=idx, radius=1)
        group_type, group_atoms = fg.identify_functional_group(mol=mol, atom_indices=current_group)

        if not any(set(group_atoms) & set(atoms) for atoms in highlight_atoms):
            highlight_atoms.append(group_atoms)
            used_atoms.update(group_atoms)

    functional_group_attention = []
    for group in highlight_atoms:
        group_attention = row_means[group]
        mean_attention = np.mean(group_attention)
        group_type, _ = fg.identify_functional_group(mol, group)
        functional_group_attention.append({
            'atoms': group,
            'attention': mean_attention,
            'type': group_type
        })
    
    return functional_group_attention

def color_atoms_by_functional_group(smiles, functional_group_attention, task, save_path, name):
    from rdkit import Chem
    from rdkit.Chem import Draw
    import matplotlib.pyplot as plt
    import numpy as np
    import json, io

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
        
    with open('functional_color_dict.json', 'r') as f:
        color_dict = json.load(f)
    atom_colors = {}
    
    for group in functional_group_attention:
        group_type = group['type']
        attention = group['attention']
        
        alpha = 0.2 + 0.8 * attention
        
        color = color_dict.get(group_type, (0.5, 0.5, 0.5))
        color_with_alpha = (*color, alpha)
        
        for atom_idx in group['atoms']:
            atom_colors[atom_idx] = color_with_alpha
    plt.rcParams['font.family'] = 'Times New Roman'

    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(800, 800)
    options = drawer.drawOptions()
    options.addAtomIndices = True
    
    drawer.DrawMolecule(mol, highlightAtoms=list(atom_colors.keys()),
                       highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    
    from IPython.display import Image

    plt.figure(figsize=(20, 12))
    plt.imshow(plt.imread(io.BytesIO(drawer.GetDrawingText())))
    plt.axis('off')
   
    for i, item in enumerate(functional_group_attention):
        y_position = 0.05 + i * 0.05  
        plt.figtext(
            0.5, y_position, 
            f"Atoms:  {item['atoms']}, Attention:  {item['attention']:.3f}, Type:  {item['type']}", 
            ha="center", fontsize=18, fontweight="bold",
            bbox={"facecolor":"orange", "alpha":0.2, "pad":5}
        )
    
    
    plt.savefig(f'{save_path}\\{name}_{task}_fg_attention.tiff', 
                format='tiff',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0)
    plt.show()
    

    return Image(drawer.GetDrawingText())

def draw_attn_pic(smiles, weights, task, save_path, name):
    import io
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import matplotlib.pyplot as plt
    import numpy as np
    
    mol = Chem.MolFromSmiles(smiles)
    first_mol_attention = torch.diagonal(weights, dim1=1, dim2=2).transpose(0, 1).squeeze().cpu().detach().numpy()
    
    mask = first_mol_attention != 0
    row_means = np.sum(first_mol_attention, axis=1)  / np.count_nonzero(mask, axis=1)
    threshold = (row_means.max() - row_means.min()) * -1e-10
    mask = row_means >= threshold
    indices = np.where(mask)[0]
    
    target_atoms = [int(idx) for idx in indices]
    
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
    
    start_color = hex_to_rgb('#ffffff')
    mid_color_1 = hex_to_rgb('#64aad0')
    mid_color_2 = hex_to_rgb('#3c9dd0')
    end_color = hex_to_rgb('#086ca2')     
    
    colors = [start_color, mid_color_1, mid_color_2, end_color]
    
    custom_cmap = LinearSegmentedColormap.from_list('enhanced_green_gradient', colors)
    
    atom_colors = {}
    for atom_idx in target_atoms:
        attention_value = row_means[atom_idx]
        color = custom_cmap(attention_value)
        atom_colors[atom_idx] = color

    d = Draw.rdDepictor.Compute2DCoords(mol)
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(800, 800)
    drawer.drawOptions().addAtomIndices = False
    drawer.drawOptions().atomHighlightsAreCircles = True

    drawer.DrawMolecule(
        mol,
        highlightAtoms=target_atoms,
        highlightAtomColors=atom_colors
    )
    drawer.FinishDrawing()

    from IPython.display import Image

    plt.figure(figsize=(10, 8))
    gs = plt.GridSpec(1, 2, width_ratios=[1, 0.03])
    ax1 = plt.subplot(gs[0])

    img = Image(drawer.GetDrawingText())
    ax1.imshow(plt.imread(io.BytesIO(drawer.GetDrawingText())))
    ax1.axis('off')

    cax = plt.axes([0.85, 0.3, 0.03, 0.4])
    norm = Normalize(vmin=0, vmax=0.6)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(cmap=custom_cmap, norm= norm),
        cax=cax,
        label='Attention Weight'
    )
 
    cb.set_label('Attention Weight', font='Times New Roman', fontsize=20, fontweight='bold')
    cb.set_ticks([0, 0.6])
    for label in cb.ax.get_yticklabels():
        label.set_font('Times New Roman')
        label.set_fontsize(18)
        label.set_fontweight('bold')

    plt.savefig(f'{save_path}\\{name}_{task}_atoms_attention.tiff', 
                format='tiff',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0)
    plt.show()

def visual_attn(smiles, decoder_path):
    for file in tqdm(os.listdir(decoder_path)):
        import ruamel.yaml as yaml
        model_path = os.path.join(decoder_path, file, 'model.pth')

        save_path = f'2d_pic'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        task = file.split('_')[0]

        name = 'geompnn'
        yaml = yaml.YAML(typ='rt')
        config = yaml.load(open(f'configs/{name}.yaml'))

        bert_config = BertConfig.from_json_file(config['bert_config'])
        tokenizer = BertTokenizer.from_pretrained(r'./tokenizer_cased')
        text_encoder = BertForMaskedLM(config= bert_config)
        model = ALBEF(text_encoder= text_encoder, tokenizer= tokenizer, config= config).to(device= 'cuda')
        load_checkpoint(model, model_path)

        input_train, label_train = graphs_from_smiles(smiles)
        train_dataset = MPNNDataset(input_train, label_train)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=prepare_batch)

        for batch in train_dataloader:
            (atom_features, bond_features, pair_indices, molecule_indicator, 
                bond_angles_features, dihedral_angles_features, bond_angle_pair_indices, 
                dihedral_angle_pair_indices, smi) = batch

        x = model.visual_encoder.message_passing(atom_features = atom_features.to(device= 'cuda'), 
                                         bond_features = bond_features.to(device= 'cuda'), 
                                         pair_indices = pair_indices.to(device= 'cuda'), 
                                         bond_angle_features = bond_angles_features.to(device= 'cuda'), 
                                         bond_angle_pair_indices = bond_angle_pair_indices.to(device= 'cuda'), 
                                         dihedral_angle_features = dihedral_angles_features.to(device= 'cuda'), 
                                         dihedral_angle_pair_indices = dihedral_angle_pair_indices.to(device= 'cuda'))
        x = model.visual_encoder.transformer_encoder.partition_padding([x, molecule_indicator.to(device= 'cuda')])

        padding_mask = (x.sum(dim= -1) != 0)

        attention_output, weights = model.visual_encoder.transformer_encoder.attention(x, x, x, key_padding_mask= padding_mask.float(), need_weights=True, average_attn_weights=False)
        functional_group_attention = get_functional_group_attn_count(smiles, weights)
        functional_group_attention.sort(key=lambda x: x['attention'], reverse=True)
        draw_attn_pic(smiles, weights, task, save_path, name)
        img = color_atoms_by_functional_group(smiles, functional_group_attention, task, save_path, name)
