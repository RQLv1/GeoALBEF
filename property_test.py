import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import ruamel.yaml as yaml
import torch.backends.cudnn as cudnn
import random, argparse

from dataloader.geo_data_utils import graphs_from_smiles, molecule_from_smiles, graph_from_molecule
from dataloader.geo_dataset import MPNNDataset, prepare_batch
from models.model_pretrain import ALBEF
from models.tokenization_bert import BertTokenizer
from models.xbert import BertForMaskedLM, BertConfig
from utils import create_logger, save_checkpoint, load_checkpoint

def get_dataloader(data_path):
    df_polymer = pd.read_csv(data_path)

    batch_size = 32
    permuted_indices = np.random.permutation(np.arange(df_polymer.shape[0]))
    train_index = permuted_indices[: int(df_polymer.shape[0] * 0.8 / batch_size) * batch_size]
    valid_index = permuted_indices[int(df_polymer.shape[0] * 0.8 / batch_size) * batch_size : int(df_polymer.shape[0] * 0.9 / batch_size) * batch_size]
    test_index = permuted_indices[int(df_polymer.shape[0] * 0.9 / batch_size) * batch_size :]

    df_train = df_polymer.iloc[train_index]
    df_valid = df_polymer.iloc[valid_index]
    df_test = df_polymer.iloc[test_index]

    input_train, label_train = graphs_from_smiles(df_train)
    train_dataset = MPNNDataset(input_train, label_train)
    total_size = len(train_dataset)
    remainder = total_size % batch_size
    if remainder != 0:
        new_size = total_size - remainder
        train_dataset = torch.utils.data.Subset(train_dataset, range(new_size))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=prepare_batch)
    torch.save(train_dataloader, 'train_dataloader.pth')

    input_valid, label_valid = graphs_from_smiles(df_valid)
    valid_dataset = MPNNDataset(input_valid, label_valid)
    total_size = len(valid_dataset)
    remainder = total_size % batch_size
    if remainder != 0:
        new_size = total_size - remainder
        valid_dataset = torch.utils.data.Subset(valid_dataset, range(new_size))
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=prepare_batch)
    torch.save(valid_dataloader, 'valid_dataloader.pth')

    input_test, label_test = graphs_from_smiles(df_test)
    test_dataset = MPNNDataset(input_test, label_test)
    total_size = len(test_dataset)
    remainder = total_size % batch_size
    if remainder != 0:
        new_size = total_size - remainder
        test_dataset = torch.utils.data.Subset(test_dataset, range(new_size))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=prepare_batch)
    torch.save(test_dataloader, 'test_dataloader.pth')

    return train_dataloader, valid_dataloader, test_dataloader
   
def main(seed, pretrain_path, model, mode='finetune'):
    seed = 42  #[42, 123, 1234]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    logger = create_logger(logger_file_path='property_prediction_logs')
    logger.info('Start training for all properties...')

    for num, task in list(property_dict.items()):
        logger.info(f'\nStarting training for property: {task}')

        decoder = Decoder(model)
        if mode == 'finetune':
            print('start finetune')
            optimizer = torch.optim.AdamW(params= decoder.parameters(), lr= 1e-3, weight_decay= 1e-4)
        else:
            print('start pretrain')
            optimizer = torch.optim.AdamW(params= decoder.fc.parameters(), lr= 1e-3, weight_decay= 1e-4)

        criterion = torch.nn.MSELoss()
        load_checkpoint(decoder, optimizer, pretrain_path)

        train_dataloader = torch.load('train_dataloader.pth')
        valid_dataloader = torch.load('valid_dataloader.pth')
        test_dataloader = torch.load('test_dataloader.pth')

        batch_size = config['batch_size']

        Valid_loss = []
        for epoch in range(8):
            try:

                #Train
                decoder.train()
                loss = []
                progress_bar = tqdm(enumerate(train_dataloader), 
                                      total=len(train_dataloader),
                                      desc=f'Epoch {epoch}',
                                      leave=False)
                for i, batch in progress_bar:

                    (atom_features, bond_features, pair_indices, molecule_indicator, 
                    bond_angles_features, dihedral_angles_features, bond_angle_pair_indices, dihedral_angle_pair_indices, smiles), \
                    label = batch
                    property_label = torch.tensor(label[num]).unsqueeze(1).to(device= 'cuda')
                    logits = decoder(atom_features, bond_features, pair_indices, molecule_indicator, bond_angles_features, 
                                    dihedral_angles_features, bond_angle_pair_indices, dihedral_angle_pair_indices, smiles)

                    l = criterion(logits, property_label)

                    progress_bar.set_postfix({
                            'loss_rmse': f'{torch.sqrt(l).item()}',})

                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
                    loss.append(torch.sqrt(l).item())
                logger.info(f"Epoch: {epoch}, Train_loss_rmse_per: {np.mean(np.array(loss))}")

                #Valid
                decoder.eval()
                eloss = []
                progress_bar = tqdm(enumerate(valid_dataloader), 
                                      total=len(valid_dataloader),
                                      desc=f'Epoch {epoch}',
                                      leave=False)
                with torch.no_grad():
                    for i, batch in progress_bar:

                        (atom_features, bond_features, pair_indices, molecule_indicator, 
                        bond_angles_features, dihedral_angles_features, bond_angle_pair_indices, dihedral_angle_pair_indices, smiles), \
                        label = batch
                        property_label = torch.tensor(label[num]).unsqueeze(1).to(device= 'cuda')
                        logits = decoder(atom_features, bond_features, pair_indices, molecule_indicator, bond_angles_features, 
                                        dihedral_angles_features, bond_angle_pair_indices, dihedral_angle_pair_indices, smiles)

                        l = criterion(logits, property_label)

                        progress_bar.set_postfix({
                                'loss_rmse': f'{torch.sqrt(l).item()}',})

                        eloss.append(torch.sqrt(l).item())
                    logger.info(f"Epoch: {epoch}, Valid_loss_rmse_per: {np.mean(np.array(eloss))}")


                if Valid_loss == []:
                    # save_checkpoint(decoder.model, optimizer, epoch, f'{task}')
                    Valid_loss.append(np.mean(np.array(eloss)))

                else:
                    if np.mean(np.array(eloss)) < min(Valid_loss):
                        # save_checkpoint(decoder.model, optimizer, epoch, f'{task}')
                        Valid_loss.append(np.mean(np.array(eloss)))
                    else:
                        Valid_loss.append(np.mean(np.array(eloss)))

                #Test
                decoder.eval()
                testloss = []
                progress_bar = tqdm(enumerate(test_dataloader), 
                                      total=len(test_dataloader),
                                      desc=f'Epoch {epoch}',
                                      leave=False)
                with torch.no_grad():
                    for i, batch in progress_bar:

                        (atom_features, bond_features, pair_indices, molecule_indicator, 
                        bond_angles_features, dihedral_angles_features, bond_angle_pair_indices, dihedral_angle_pair_indices, smiles), \
                        label = batch
                        property_label = torch.tensor(label[num]).unsqueeze(1).to(device= 'cuda')
                        logits = decoder(atom_features, bond_features, pair_indices, molecule_indicator, bond_angles_features, 
                                        dihedral_angles_features, bond_angle_pair_indices, dihedral_angle_pair_indices, smiles)

                        l = criterion(logits, property_label)

                        progress_bar.set_postfix({
                                'loss_rmse': f'{torch.sqrt(l).item()}',})

                        testloss.append(torch.sqrt(l).item())
                    logger.info(f"Epoch: {epoch}, Test_loss_rmse_per: {np.mean(np.array(testloss))}")
            except:
                continue
            torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type= str, default= 'geoalbef')
    parser.add_argument('--seed', type= int, default= 42)
    parser.add_argument('--pretrain_path', type= str, default= 'geomodel.pth')
    parser.add_argument('--mode', type= str, default= 'finetune')
    parser.add_argument('--data_path', type= str, default= 'mod_data.csv')
    parser.add_argument('--is_get_dataloader', type=lambda x: x.lower() == 'true', default= True)
    args = parser.parse_args()

    if args.is_get_dataloader:
        print('start get dataloader')
        _, _, _ = get_dataloader(args.data_path)

    property_dict = {0: 'Egc', 1: 'Egb', 2: 'Eib', 3: 'Ei', 4: 'Eea', 5: 'nc', 6: 'ne',
                    7: 'TSb', 8: 'TSy', 9: 'YM', 10: 'permCH4', 11: 'permCO2', 12: 'permH2', 
                    13: 'permO2', 14: 'permN2', 15: 'permHe', 16: 'Eat', 17: 'LOI', 18: 'Xc', 
                    19: 'Xe', 20: 'Cp', 21: 'Td', 22: 'Tg', 23: 'Tm'}

    p2n_dict = {}
    for i, p in property_dict.items():
        p2n_dict[p] = i

    name = args.name
    yaml = yaml.YAML(typ='rt')
    config = yaml.load(open(f'configs/{name}.yaml'))
    print('config load!')

    bert_config = BertConfig.from_json_file(config['bert_config'])
    tokenizer = BertTokenizer.from_pretrained(r'./tokenizer_cased')
    text_encoder = BertForMaskedLM(config= bert_config)
    model = ALBEF(text_encoder= text_encoder, tokenizer= tokenizer, config= config).to(device= 'cuda')

    class Decoder(nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model
            self.fc = nn.Linear(768, 1).to(device='cuda')

        def forward(self, atom_features, 
                          bond_features, 
                          pair_indices, 
                          molecule_indicator, 
                          bond_angles_features, 
                          dihedral_angles_features, 
                          bond_angle_pair_indices, 
                          dihedral_angle_pair_indices, 
                          smiles):

            image_embeds = model.visual_encoder(atom_features=atom_features.to(device= 'cuda'),
                                                bond_features=bond_features.to(device= 'cuda'), 
                                                pair_indices=pair_indices.to(device= 'cuda'),
                                                molecule_indicator=molecule_indicator.to(device= 'cuda'),
                                                bond_angle_features=bond_angles_features.to(device= 'cuda'),
                                                bond_angle_pair_indices=bond_angle_pair_indices.to(device= 'cuda'),
                                                dihedral_angle_features=dihedral_angles_features.to(device= 'cuda'),
                                                dihedral_angle_pair_indices=dihedral_angle_pair_indices.to(device= 'cuda'))

            out = self.fc(image_embeds[:, 0, :])
            return out

    print('start property predict!')
    main(args.seed, args.pretrain_path, model, args.mode)
