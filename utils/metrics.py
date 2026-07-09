import pandas as pd
import numpy as np
import os
from train import Trainer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics as skmet
import plotly.graph_objects as go
from scipy import stats

def make_all_predictions(save_path, data):

    df = pd.DataFrame(data)

    df.to_csv(os.path.join(save_path, 'all_preds_best_checkpoint.csv'), index=False)

def plot_loss_history(save_path, val_loss, train_loss):

    epochs = [i + 1 for i in range(len(train_loss))]

    plt.plot(epochs, train_loss, color='r', linestyle='solid')
    plt.plot(epochs, val_loss, color='b', linestyle='solid')
    plt.grid(color='black', linestyle='dotted', linewidth=0.7)
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss throughout epochs")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss_history.png"), dpi=300)

    plt.figure()


def auc_and_roc_curve (path_preds=None, save_path=None):

    df = pd.read_csv(os.path.join(path_preds, 'all_preds_best_checkpoint.csv'))

    lab_real, lab_pred = df['labels'].to_numpy(), df['preds'].to_numpy()

    fpr, tpr, thresholds = skmet.roc_curve(lab_real, lab_pred)
    auc = skmet.roc_auc_score(lab_real, lab_pred)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines+markers",
        hovertemplate=
            "Threshold: %{customdata:.2f}<br>" +
            "TPR: %{y:.2f}<br>" +
            "FPR: %{x:.2f}<extra></extra>",
        customdata=thresholds,
        name=f"ROC (AUC = {auc:.3f})"
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(dash="dash"),
        name="Random"
    ))

    fig.update_layout(
        title="Curva ROC interativa",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)"
    )

    fig.write_html(os.path.join(save_path, "roc_interativa.html"))

    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC (AUC = {auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        linewidth=1.5,
        label="Random"
    )

    plt.xlim([0, 1])
    plt.ylim([0, 1.05])

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_path, "roc_curve.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    #return auc

def plot_histogram_comparison(path_preds, save_path, file_name):

    df_high = pd.read_csv(os.path.join(path_preds, 'preds_high.csv'))
    df_low = pd.read_csv(os.path.join(path_preds, 'preds_low.csv'))

    good_preds = df_high['preds'].to_numpy()
    bad_preds = df_low['preds'].to_numpy()

    plt.figure()
    plt.hist(good_preds, label="BOAS", color="blue", alpha=0.7)
    plt.hist(bad_preds, label="RUINS", color="orange", alpha=0.7)

    plt.title('Histogram of Predictions Scores')
    plt.xlabel('Prediction Scores')
    plt.ylabel('Number of Images')

    plt.legend(loc='upper right')

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()

def separate_preds_high_low(path_all_preds, save_path, mode='train'):
    df = pd.read_csv(path_all_preds)

    preds_high = df[df['labels'] == 1].sort_values(['preds'], ascending=False)
    preds_low = df[df['labels'] == 0].sort_values(['preds'], ascending=False)

    preds_high.to_csv(os.path.join(save_path, 'preds_high.csv'), index=False)
    preds_low.to_csv(os.path.join(save_path, 'preds_low.csv'), index=False)

def get_threshold(experiment_path):

    preds_high = pd.read_csv(os.path.join(experiment_path, 'preds_high.csv'))

    threshold_at_95 = preds_high['preds'].quantile(0.05)

    data = {

        'threshold_at_95': [threshold_at_95]
    }

    df = pd.DataFrame(data)

    df.to_csv(os.path.join(experiment_path, 'threshold_at_95.csv'), index=False)

def calculate_tpr_tnr(experiment_path):

    preds_high = pd.read_csv(os.path.join(experiment_path, 'test', 'preds_high.csv'))
    preds_low = pd.read_csv(os.path.join(experiment_path, 'test', 'preds_low.csv'))

    threshold_at_95 = pd.read_csv(os.path.join(experiment_path, 'threshold_at_95.csv'))['threshold_at_95'][0]

    tpr = (preds_high['preds'] >= threshold_at_95).sum() / len(preds_high)
    tnr = (preds_low['preds'] < threshold_at_95).sum() / len(preds_low)

    data = {
        'TPR': [tpr],
        'TNR': [tnr]
    }

    df = pd.DataFrame(data)

    df.to_csv(os.path.join(experiment_path, 'test', 'tpr_tnr.csv'), index=False)

def calculate_srocc_and_plcc(x, y, save_path):

    x = np.array(x)
    y = np.array(y)

    srocc, _ = stats.spearmanr(x, y)
    plcc = np.corrcoef(x, y)[0, 1]

    data = {
        'SROCC': [srocc],
        'PLCC': [plcc]
    }

    df = pd.DataFrame(data)

    df.to_csv(os.path.join(save_path, 'metrics.csv'), index=False)

def calculate_all_metrics(save_path, trainer: Trainer):

    data = {
        'img': trainer.imgs_id,
        'preds': trainer.preds,
        'mos': trainer.scores,
        'labels': trainer.labels
    }

    calculate_srocc_and_plcc(data['mos'], data['preds'], save_path)

    make_all_predictions(save_path, data)

    separate_preds_high_low(os.path.join(save_path, 'all_preds_best_checkpoint.csv'), save_path)

    get_threshold(save_path)

    plot_loss_history(save_path, trainer.loss_history_eval, trainer.loss_history_train)

    plot_histogram_comparison(path_preds=save_path, save_path=save_path, file_name='histogram.png')

    auc_and_roc_curve (path_preds=save_path, save_path=save_path)


def agg_metrics(path, num_folders):
    
    """
    Esta função agrega as métricas de todas as folders de determinado experimento. Salva os resultados agregados no
    caminho passado como parâmetro.

    :param ex_path(string): caminho para a pasta onde estão salvos os resultados dos experimentos.
    :param num_folders(int): número de folders utilizado na validação cruzada.
    """

    srocc_agg, plcc_agg, threshold_at_95_agg, tpr_agg, tnr_agg, discarded_ood_agg = [], [], [], [], [], []

    for folder in range(num_folders):

        folder_path = os.path.join(path, f"folder_{folder+1}")
        
        metrics_csv_path = os.path.join(folder_path, "test/metrics.csv")
        metrics = pd.read_csv(metrics_csv_path)
        srocc = np.array(metrics["SROCC"])
        plcc = np.array(metrics["PLCC"])

        threshold_at_95_csv_path = os.path.join(folder_path, "threshold_at_95.csv")
        threshold_at_95 = pd.read_csv(threshold_at_95_csv_path)["threshold_at_95"][0]

        tpr_tnr_csv_path = os.path.join(folder_path, "test/tpr_tnr.csv")
        tpr_tnr = pd.read_csv(tpr_tnr_csv_path)
        tpr = tpr_tnr["TPR"][0]
        tnr = tpr_tnr["TNR"][0]

        ood_preds_csv_path = os.path.join(folder_path, "test/ood_preds_best_checkpoint.csv")
        ood_preds = pd.read_csv(ood_preds_csv_path)
        discarded_ood = (ood_preds['preds'] < threshold_at_95).sum() / len(ood_preds)

        srocc_agg.append(srocc)
        plcc_agg.append(plcc)
        threshold_at_95_agg.append(threshold_at_95)
        tpr_agg.append(tpr)
        tnr_agg.append(tnr)
        discarded_ood_agg.append(discarded_ood)

    metrics = {
        'srocc': [f'{np.mean(srocc_agg):.4f} +- {np.std(srocc_agg):.4f}'],
        'plcc': [f'{np.mean(plcc_agg):.4f} +- {np.std(plcc_agg):.4f}'],
        'threshold_at_95': [f'{np.mean(threshold_at_95_agg):.4f} +- {np.std(threshold_at_95_agg):.4f}'],
        'tpr': [f'{np.mean(tpr_agg):.4f} +- {np.std(tpr_agg):.4f}'],
        'tnr': [f'{np.mean(tnr_agg):.4f} +- {np.std(tnr_agg):.4f}'],
        'discarded_ood': [f'{np.mean(discarded_ood_agg):.4f} +- {np.std(discarded_ood_agg):.4f}']
    }

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(os.path.join(path, "agg_metrics.csv"), index=False)