# GeoALBEF
## About
This repository contains the code and resources of the following paper:

Multimodal modeling for polymer property prediction

## Overview of the framework
By integrating the chemical semantic sequence information with the intrinsic and spatial information of polymers, a deep neural network architecture with cross-modal interaction capability was developed. 

<p align="center">
<img  src="toc.png"> 
</p>

## **Setup environment**
Setup the required environment using `environment.yml` with Anaconda. While in the project directory run:

    conda env create -f environment.yml

Activate the environment

    conda activate geoalbef

## **Pre-trained model and data download**
data/smiles.json train_dataloader.pth valid_dataloader.pth test_dataloader.pth geomodel.pth dataloader.pth downloaded from https://zenodo.org/uploads/14177593
Place the data and pre-training files in the following directory structure

GeoALBEF-main/  
├── data/  
│ ├── smiles.json   
├── configs/     
├── dataloader/  
├── models/  
├── tokenizer_cased/    
├── visual/     
├── train_dataloader.pth    
├── valid_dataloader.pth    
├── test_dataloader.pth  
├── dataloader.pth  
├── geomodel.pth    
├── mod_data.csv    
├── property_test   
├── train.py    
├── utils   
├── README.md  
└── ...              

## **Pre-train bond length enhanced BERT**

Execute the command:

    cd your_download_path\GeoALBEF-main
    python train.py --name geoalbef --dataloader_path dataloader.pth --is_get_dataloader false

## **Pre-train GeoALBEF**

Execute the command:

    cd your_download_path\GeoALBEF-main
    python property_test.py --name geoalbef --seed 42 --pretrain_path geomodel.pth --mode pretrain --data_path mod_data.csv --is_get_dataloader false

## **Finetune the GeoALBEF**

Execute the command:

    cd your_download_path\GeoALBEF-main
    python property_test.py --name geoalbef --seed 42 --pretrain_path geomodel.pth --mode finetune --data_path mod_data.csv --is_get_dataloader false

## **Structure-property plot**
Fine-tune the model to get the optimal model saved under each property .pth file or download from https://zenodo.org/records/14958954

Save it in the following directory:

GeoALBEF-main/  
├── data/  
│ ├── smiles.json   
├── configs/     
├── dataloader/  
├── models/  
├── tokenizer_cased/    
├── visual/  
│ ├── only_model_ckp/  
│ │ ├── only_model_ckp/  
│ │ │ ├── ne_checkpoints/    
│ │ │ │ ├── model.pth  
└── ... 

Execute the command:

    cd your_download_path\GeoALBEF-main\visual

    from visual import visual_attn

    decoder_path = r'only_model_ckp'
    smiles = '[*]CCCn1ccc(-c2ccn(CCC[*])n2)n1'

    visual_attn(smiles, decoder_path)

Get the image in the 2d_pic folder:
<p align="center">
<img  src="geoalbef_Eea_fg_attention.png"> 
</p>
<p align="center">
<img  src="geoalbef_Eea_atoms_attention.png"> 
</p>

    
## License
GeoALBEF is licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0.

## Acknowledgments and Third-Party Code Statements
This project uses code from [ALBEF](https://github.com/salesforce/ALBEF?tab=BSD-3-Clause-1-ov-file) (BSD-3-Clause protocol) with modifications.
The original code is copyrighted to the authors of ALBEF, and the modifications are copyrighted to [RQLv].