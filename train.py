import warnings
warnings.simplefilter('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import argparse, json
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import random
import numpy as np
import ruamel.yaml as yaml
from transformers import get_cosine_schedule_with_warmup

from models.model_pretrain import ALBEF
from models.tokenization_bert import BertTokenizer
from models.xbert import BertForMaskedLM, BertConfig

from dataloader.geo_dataset import MPNNDataset, prepare_batch
from torch.utils.data import DataLoader
from dataloader.geo_data_train import graphs_from_smiles

from utils import create_logger, load_checkpoint, save_checkpoint

def get_dataloader(config):
    smiles_file = config['smiles_file']
    with open(smiles_file, 'r') as f:
        smiles_dict = json.load(f)
    smiles_list = list(smiles_dict.values())
    input, smiles_list_checked = graphs_from_smiles(smiles_list)
    dataset = MPNNDataset(input, smiles_list_checked)
    batch_size = config['batch_size']
    total_size = len(dataset)
    remainder = total_size % batch_size
    if remainder != 0:
        new_size = total_size - remainder
        dataset = torch.utils.data.Subset(dataset, range(new_size))
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=prepare_batch)
    torch.save(dataloader, 'dataloader.pth')
    return dataloader

def train(model, tokenizer, dataloader, name, config):
    
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    num_epochs = 20
    num_training_steps = num_epochs * len(dataloader)
    num_warmup_steps = num_training_steps // 20 #5% warmup

    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer= optimizer,
        num_warmup_steps= num_warmup_steps,
        num_training_steps= num_training_steps,
    )
    min_lr = config['min_lr']

    logger = create_logger(logger_file_path= 'logs')
    logger.info('Start training...')
    
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path=f'{name}_checkpoints/Epoch_{config["start_epoch"]}')

    for epoch in tqdm(range(num_epochs)):
        model.train()
        progress_bar = tqdm(enumerate(dataloader), 
                          total=len(dataloader),
                          desc=f'Epoch {epoch}',
                          leave=False)
        for i, (batch_data, smiles) in progress_bar:

            (atom_features, bond_features, pair_indices, 
             molecule_indicator, bond_angle_features,
             dihedral_angle_features, bond_angle_pair_indices, 
             dihedral_angle_pair_indices) = batch_data
            
            optimizer.zero_grad()
            atom_features = atom_features.to(device= 'cuda', non_blocking= True)
            bond_features = bond_features.to(device= 'cuda', non_blocking= True)
            pair_indices = pair_indices.to(device= 'cuda', non_blocking= True)
            molecule_indicator = molecule_indicator.to(device= 'cuda', non_blocking= True)
            bond_angle_features = bond_angle_features.to(device= 'cuda', non_blocking= True)
            dihedral_angle_features = dihedral_angle_features.to(device= 'cuda', non_blocking= True)
            bond_angle_pair_indices = bond_angle_pair_indices.to(device= 'cuda', non_blocking= True)
            dihedral_angle_pair_indices = dihedral_angle_pair_indices.to(device= 'cuda', non_blocking= True)
            text_input = tokenizer(smiles, padding= 'longest', truncation= True, max_length= 256, return_tensors= 'pt').to(device= 'cuda')
            
            if epoch > 0:
                alpha = config['alpha']
            else:
                alpha = config['alpha'] * min(1, i/len(dataloader))
    
            loss_mlm, loss_itc, loss_itm = model(atom_features=atom_features,
                                                bond_features=bond_features, 
                                                pair_indices=pair_indices,
                                                molecule_indicator=molecule_indicator,
                                                bond_angle_features=bond_angle_features,
                                                bond_angle_pair_indices=bond_angle_pair_indices,
                                                dihedral_angle_features=dihedral_angle_features,
                                                dihedral_angle_pair_indices=dihedral_angle_pair_indices, 
                                                text= text_input, alpha= alpha)
            loss = loss_mlm + loss_itc + loss_itm
            
            loss.backward()
            optimizer.step()
            
            scheduler.step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], min_lr)

            progress_bar.set_postfix({
                'loss_mlm': f'{loss_mlm.item():.4f}',
                'loss_itc': f'{loss_itc.item():.4f}', 
                'loss_itm': f'{loss_itm.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
    
            if (i + 1) % 3739 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f'Epoch {epoch}, Step: {i}')
                logger.info(f'loss_mlm: {loss_mlm.item()}')
                logger.info(f'loss_itc: {loss_itc.item()}')
                logger.info(f'loss_itm: {loss_itm.item()}')
                logger.info(f'loss: {loss.item()}')
                logger.info(f'LR: {current_lr: .2e}')
        
            del atom_features, bond_features, pair_indices, molecule_indicator, \
                bond_angle_features, dihedral_angle_features, bond_angle_pair_indices, \
                    dihedral_angle_pair_indices, text_input, loss_mlm, loss, smiles
            torch.cuda.empty_cache()
        save_checkpoint(model, optimizer, epoch, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type= str, default= 'geoalbef')
    parser.add_argument('--dataloader_path', type= str, default= 'dataloader.pth')
    parser.add_argument('--is_get_dataloader', type=lambda x: x.lower() == 'true', default= True)
    args = parser.parse_args()

    name = args.name
    dataloader_path = args.dataloader_path
    is_get_dataloader = args.is_get_dataloader

    print("dataloader & config loading...")
    yaml = yaml.YAML(typ='rt')
    config = yaml.load(open(f'configs/{name}.yaml'))

    if is_get_dataloader:
        dataloader = get_dataloader(config)
        print('dataloader get!')
    else:
        dataloader = torch.load(dataloader_path)
        print('dataloader load!')

    print("start training......")

    bert_config = BertConfig.from_json_file(config['bert_config'])
    tokenizer = BertTokenizer.from_pretrained(r'./tokenizer_cased')
    text_encoder = BertForMaskedLM(config= bert_config)
    model = ALBEF(text_encoder= text_encoder, tokenizer= tokenizer, config= config)
    model.to(device= 'cuda')
    train(model, tokenizer, dataloader, name, config)
