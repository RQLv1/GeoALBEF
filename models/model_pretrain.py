# Copyright (c) 2021, Salesforce.com, Inc.
# All rights reserved.
# This file is based on ALBEF code modified to follow the BSD-3-Clause license.

from models.geo_mpnn import MPNNModel
import torch
import torch.nn.functional as F
from torch import nn


class ALBEF(nn.Module):
    def __init__(self,
                 text_encoder= None,
                 tokenizer= None,
                 config= None,):
        super().__init__()

        self.tokenizer = tokenizer
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']


        vision_width = config['vision_width']
        self.visual_encoder = MPNNModel(atom_dim = config['atom_dim'], 
                                        bond_dim = config['bond_dim'], 
                                        bond_angle_dim = config['bond_angle_dim'], 
                                        dihedral_angle_dim = config['dihedral_angle_dim'], 
                                        batch_size= config['batch_size'], 
                                        message_units= 88, 
                                        message_steps=5, 
                                        num_attention_heads= 8, 
                                        dense_units= vision_width)
        
        
        self.text_encoder = text_encoder
        text_width = self.text_encoder.config.hidden_size #768

      
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size'] 
        self.mometum = config['momentum'] 
        self.itm_head = nn.Linear(text_width, 2) 

        self.visual_encoder_m = MPNNModel(atom_dim = config['atom_dim'], 
                                        bond_dim = config['bond_dim'], 
                                        bond_angle_dim = config['bond_angle_dim'], 
                                        dihedral_angle_dim = config['dihedral_angle_dim'], 
                                        batch_size= config['batch_size'], 
                                        message_units= 88, 
                                        message_steps=5, 
                                        num_attention_heads= 8, 
                                        dense_units= vision_width)
        
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = text_encoder
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]
        self.copy_params() 


        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype= torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim= 0)
        self.text_queue = nn.functional.normalize(self.text_queue,dim= 0)
    
    def forward(self, 
                atom_features, 
                bond_features, 
                pair_indices, 
                molecule_indicator, 
                bond_angle_features, 
                dihedral_angle_features, 
                bond_angle_pair_indices, 
                dihedral_angle_pair_indices, 
                text, 
                alpha= 0):
        with torch.no_grad():
            self.temp.clamp(0.001, 0.5)


        image_embeds = self.visual_encoder(atom_features=atom_features,
                                            bond_features=bond_features, 
                                            pair_indices=pair_indices,
                                            molecule_indicator=molecule_indicator,
                                            bond_angle_features=bond_angle_features,
                                            bond_angle_pair_indices=bond_angle_pair_indices,
                                            dihedral_angle_features=dihedral_angle_features,
                                            dihedral_angle_pair_indices=dihedral_angle_pair_indices)
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype= torch.long).to(atom_features.device)
        image_feat = self.vision_proj(image_embeds[:, 0, :])
        image_feat = F.normalize(image_feat, dim= -1)

        text_output = self.text_encoder.bert(text.input_ids, attention_mask= text.attention_mask, 
                                             return_dict= True, mode= 'text')
        text_embeds = text_output.last_hidden_state
        text_feat = self.text_proj(text_embeds[:, 0, :])
        text_feat = F.normalize(text_feat, dim= -1) 

        ###========================ITA============================###

        with torch.no_grad():
            self._momentum_update() 

            image_embeds_m = self.visual_encoder_m(atom_features=atom_features,
                                            bond_features=bond_features, 
                                            pair_indices=pair_indices,
                                            molecule_indicator=molecule_indicator,
                                            bond_angle_features=bond_angle_features,
                                            bond_angle_pair_indices=bond_angle_pair_indices,
                                            dihedral_angle_features=dihedral_angle_features,
                                            dihedral_angle_pair_indices=dihedral_angle_pair_indices)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]),dim =-1)

            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim= 1)

            text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask = text.attention_mask,
                                                     return_dict= True, mode= 'text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim= -1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim= -1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp 

            sim_targets = torch.zeros(sim_i2t_m.size()).to(atom_features.device)
            sim_targets.fill_diagonal_(1) 

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim= 1) + (1-alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim= 1) + (1-alpha) * sim_targets
       
        sim_i2t = image_feat @ text_feat_all / self.temp 
        sim_t2i = text_feat @ image_feat_all / self.temp 

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim= 1) * sim_i2t_targets, dim= 1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim= 1) * sim_t2i_targets, dim= 1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        
        ###========================ITM============================###
       
        output_pos = self.text_encoder.bert(
            encoder_embeds= text_embeds,
            attention_mask= text.attention_mask,
            encoder_hidden_states= image_embeds,
            encoder_attention_mask= image_atts,
            return_dict= True,
            mode= 'fusion',
        )
        with torch.no_grad():
            bs = image_embeds.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim= 1) 
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim= 1) 

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0) 

        
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item() 
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim= 0)

        
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim= 0)
        text_atts_neg = torch.stack(text_atts_neg, dim= 0)


        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim= 0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim= 0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim= 0)
        image_atts_all = torch.cat([image_atts, image_atts], dim= 0)

        output_neg = self.text_encoder.bert(
            encoder_embeds = text_embeds_all,
            attention_mask= text_atts_all,
            encoder_hidden_states= image_embeds_all,
            encoder_attention_mask= image_atts_all,
            return_dict= True,
            mode= 'fusion',
        )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim= 0) 
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype= torch.long), torch.zeros(2*bs, dtype= torch.long)],
                               dim= 0).to(atom_features.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        ###=========================MLM=========================###
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, atom_features.device,
                                      targets= labels, probability_matrix= probability_matrix)
        

        with torch.no_grad():
            logits_m = self.text_encoder_m(input_ids,
                                           attention_mask= text.attention_mask,
                                           encoder_hidden_states = image_embeds_m,
                                           encoder_attention_mask = image_atts,
                                           return_dict = True,
                                           return_logits= True,)

        mlm_output = self.text_encoder(input_ids,
                                           attention_mask= text.attention_mask,
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,
                                           return_dict = True,
                                           labels= labels,
                                           soft_labels = F.softmax(logits_m, dim= -1),
                                           alpha= alpha
                                           )
        loss_mlm = mlm_output.loss

        return loss_mlm, loss_ita, loss_itm


    @torch.no_grad()
    def copy_params(self):
        
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.mometum + param.data * (1. - self.mometum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        

        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        batch_size = image_feats.shape[0]


        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0 

        self.image_queue[:, ptr:ptr+batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr+batch_size] = text_feats.T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
    
@torch.no_grad()
def concat_all_gather(tensor):

    # tensors_gather = [torch.ones_like(tensor)
    #                   for _ in range(torch.distributed.get_world_size())]
    # torch.distributed.all_gather(tensors_gather, tensor, async_op= False)

    # output = torch.cat(tensors_gather, dim= 0)
    return tensor



