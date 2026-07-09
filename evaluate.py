import torch
import numpy as np
import pandas as pd
from utils.metrics import separate_preds_high_low, auc_and_roc_curve, plot_histogram_comparison, calculate_srocc_and_plcc, calculate_tpr_tnr
from tqdm import tqdm
import os

def test_evaluate(model, iterator, device):
    model.to(device)
    model.eval()
    all_preds, all_scores, all_labels, all_imgs = [], [], [], []
    with torch.no_grad():
        with tqdm(total=len(iterator), ascii=True, desc='Evaluating...: ', ncols=100) as t:
            for batch in iterator:
                x, score, label, img_id = batch[0].to(device), batch[1], batch[2], batch[3]
                output = model(x)
                output = output.squeeze(-1)
                preds = output
                all_preds.extend(preds.cpu().numpy())
                all_scores.extend(score)
                all_labels.extend(label)
                all_imgs.extend(img_id)
                t.update()
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    return {
        'preds': all_preds,
        'labels': all_labels,
        'ids': all_imgs,
        'scores': all_scores
    }

def test_another_dataset(model, device, test_dataloader, save_folder_path):


    test_data = test_evaluate(model, test_dataloader, device)

    df_dict = {
        'img_name': test_data['ids'],
        'preds': test_data['preds']
    }

    df = pd.DataFrame(df_dict)
    
    df.to_csv(os.path.join(save_folder_path, "another_dataset_preds.csv"), index=False)

def test_model(model, device, test_dataloader, ood_dataloader, save_folder_path):

    test_folder = os.path.join(save_folder_path, "test")

    if not os.path.exists(test_folder):
        os.makedirs(test_folder, exist_ok=True)

    eval_data = test_evaluate(model, test_dataloader, device)

    df_dict = {
        'img_name': eval_data['ids'],
        'mos': eval_data['scores'],
        'preds': eval_data['preds'],
        'labels': eval_data['labels']
    }

    if ood_dataloader is not None:
        ood_eval_data = test_evaluate(model, ood_dataloader, device)

        df_ood = pd.DataFrame({
            'img_name': ood_eval_data['ids'],
            'mos': ood_eval_data['scores'],
            'preds': ood_eval_data['preds'],
            'labels': ood_eval_data['labels']
        })

        df_ood.sort_values(by="preds", ascending=False, inplace=True)

        df_ood.to_csv(os.path.join(test_folder, "ood_preds_best_checkpoint.csv"), index=False)

    df = pd.DataFrame(df_dict)
    df = df.sort_values(
        by="img_name",
        key=lambda x: x.str.extract(r"(\d+)")[0].astype(int)
    )

    calculate_srocc_and_plcc(df['mos'], df['preds'], test_folder)
    df.to_csv(os.path.join(test_folder, "all_preds_best_checkpoint.csv"), index=False)
    separate_preds_high_low(path_all_preds=os.path.join(test_folder, 'all_preds_best_checkpoint.csv'), save_path=test_folder)
    calculate_tpr_tnr(save_folder_path)
    plot_histogram_comparison(path_preds=test_folder, save_path=test_folder, file_name='histogram.png')
    auc_and_roc_curve(path_preds=test_folder, save_path=test_folder)