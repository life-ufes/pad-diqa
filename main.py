import json
from utils.loader import get_data_loaders_pad_diqa, get_test_data_loader_pad_diqa, get_test_data_loader
import torch
import torch.nn as nn
from train import Trainer
import os
from utils.metrics import calculate_all_metrics
from evaluate import test_model, test_another_dataset
from sacred import Experiment
from utils.metrics import agg_metrics
from model.backbones import select_backbone_model

with open("./config.json") as json_file:
    _CONFIG = json.load(json_file)

DATASET_PATH = _CONFIG['dataset_pad_diqa'] if 'dataset_pad_diqa' in _CONFIG else None
DATASET_TEST = _CONFIG['another_dataset_test'] if 'another_dataset_test' in _CONFIG else None
OOD_PATH = _CONFIG['ood_path'] if 'ood_path' in _CONFIG else None
MODEL_CHECKPOINT_PATH = _CONFIG['model_checkpoint_path'] if 'model_checkpoint_path' in _CONFIG else None
SAVE_FOLDER_PATH = _CONFIG['save_folder_full_path']

ex = Experiment()

@ex.config
def my_config():
     _use_ood_in_test = False
     _use_another_dataset = False
     _model = 'mobilenet_v3_large'

@ex.automain
def main(_use_ood_in_test, _use_another_dataset, _model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if _use_another_dataset:

        model = select_backbone_model(_model)

        model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH))

        another_dataset = get_test_data_loader(dataset_path=DATASET_TEST, batch_size=30)
        test_another_dataset(model, device, another_dataset, SAVE_FOLDER_PATH)

    else:
        ood_test_dataloader = None

        train_dataloader_list, val_dataloader_list = get_data_loaders_pad_diqa(dataset_path=DATASET_PATH, num_folders=10, norm_score_10=False, batch_size=30)   
        
        test_dataloader = get_test_data_loader_pad_diqa(dataset_path=DATASET_PATH, norm_score_10=False, batch_size=30)
        if _use_ood_in_test:
            ood_test_dataloader = get_test_data_loader_pad_diqa(ood_path=OOD_PATH, batch_size=30)

        for i, (train_dataloader), in enumerate(train_dataloader_list):

            _save_folder_k = os.path.join(SAVE_FOLDER_PATH, f"folder_{i+1}")

            if not os.path.exists(_save_folder_k):
                os.makedirs(_save_folder_k, exist_ok=True)

            model = select_backbone_model(_model)

            learning_rate=2e-4

            criterion = nn.MSELoss()

            optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=learning_rate,
                        weight_decay=1e-5
                    )
            
            print(f"\n\n------------- STARTING TRAIN FOLDER {i+1}---------------\n\n")
        
    
            trainer = Trainer(device=device, save_name=os.path.join(_save_folder_k, 'best_checkpoint.pth'), criterion=criterion, optimizer=optimizer)

            trainer.fit(model=model , max_epochs=40, train_loader=train_dataloader, val_loader=val_dataloader_list[i])

            calculate_all_metrics(_save_folder_k, trainer=trainer)

            # test
            test_model(model, device, test_dataloader, ood_test_dataloader, _save_folder_k)

        agg_metrics(SAVE_FOLDER_PATH, 10)