"""
Utilitas pendukung Case Study 04 – Face Recognition
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import pathlib, random
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage


def plot_cm(y_true, y_pred, labels, figsize=(10,8)):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix'); plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.show()


def clf_report_df(y_true, y_pred, labels):
    """Classification report → DataFrame (untuk slide)"""
    report = classification_report(y_true, y_pred,
                                   target_names=labels,
                                   output_dict=True)
    return pd.DataFrame(report).T.round(3)


def plot_9_wrong(df_wrong, data_dir, target_size=(128,128)):
    """
    df_wrong : DataFrame(hasCols=[file,true,pred,conf])
    data_dir : root folder gambar
    """
    fig, ax = plt.subplots(3,3, figsize=(8,8))
    for i, (_, row) in enumerate(df_wrong.head(9).iterrows()):
        img_path = pathlib.Path(data_dir) / row.file
        img = kimage.load_img(img_path, target_size=target_size)
        ax.flat[i].imshow(img)
        ax.flat[i].set_title(f"{row.file.split('/')[-1]}\nTrue:{row.true}  Pred:{row.pred}  {row.conf:.2f}",
                             fontsize=8)
        ax.flat[i].axis('off')
    plt.tight_layout(); plt.show()


def save_top9_wrong(y_val, y_pred, probs, filenames, labels, data_dir,
                    out_csv='04-outputs/error_top9.csv'):
    """Simpan 9 sampel salah confidence tertinggi"""
    df = pd.DataFrame({'file':filenames,
                       'true': [labels[idx] for idx in y_val],
                       'pred': [labels[idx] for idx in y_pred],
                       'conf': np.max(probs, axis=1)})
    wrong = df[df['true']!=df['pred']].nlargest(9, 'conf')
    wrong.to_csv(out_csv, index=False)
    print(f"Top-9 error tersimpan: {out_csv}")
    return wrong