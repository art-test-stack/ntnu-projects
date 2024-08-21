from util import tile_tv_images

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def top_anomalous(k, losses, preds, labels):
    idx = np.flip(np.argsort(losses))[:k]

    new_labels = [ 
        str(f"label: {labels[_id]},\n  loss: {int(losses[_id] * 1e4) / 1e2}e-2") for _id in idx
        ]
    tile_tv_images(images=preds[idx], labels=new_labels)

def top_mean_anomalous(k, mean_accs, imgs_predicted, labels):  
    mean_labels = np.flip(np.argsort(mean_accs))[:k]
    is_color = imgs_predicted.shape[1] == 3
    new_labels = [ 
        str(f"label: {mean_label},\n  loss: {int(mean_accs[mean_label] * 1e4) / 1e2}e-2") for mean_label in mean_labels
        ]
    np_imgs_predicted = imgs_predicted.reshape(-1, 28, 28).cpu().detach().numpy() if not is_color else imgs_predicted.permute(0,2,3,1).cpu().detach().numpy()

    anomalies_pred = np.array([ np_imgs_predicted[np.argwhere(labels == mean_label)[:,0]][0] for mean_label in mean_labels ])

    tile_tv_images(images=anomalies_pred, labels=new_labels)

def errorbar_classification(classes, mean_accuracy, std_accuracy, label: str = "Loss"):
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(classes, mean_accuracy, width, yerr=std_accuracy, label='Accuracy')

    ax.set_ylabel(label)
    ax.set_title(f'{label} by class')
    ax.set_xticks(classes)
    ax.set_xticklabels(classes)
    ax.legend()

    plt.show()