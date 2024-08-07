import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import mmengine

def calculate_confusion_matrix(gt_path, pred_path, num_classes, reduce_zero_label=False):
    confusion_matrix = np.zeros(shape=[num_classes, num_classes])
    images_list = os.listdir(gt_path)
    for item in images_list:
        gt = os.path.join(gt_path, item)
        pred = os.path.join(pred_path, item)
        gt_segm = np.array(Image.open(gt)).astype(int)
        if reduce_zero_label:
            gt_segm[gt_segm==0] = 256
            gt_segm = gt_segm -1
        res_segm  = np.array(Image.open(pred)).astype(int)
        inds = num_classes * gt_segm + res_segm
        inds = inds.flatten()
        mat = np.bincount(inds, minlength=num_classes**2).reshape(num_classes, num_classes)
        confusion_matrix += mat
    return confusion_matrix




def plot_confusion_matrix(confusion_matrix,
                          name,
                          labels=['background','snow', 'ice'],
                          save_dir='/aipt/CZX/mmsegmentation_czx/work_dirs/ice_snow/confusion_matrix/',
                          show=True,
                          title='Normalized Confusion Matrix',
                          color_theme='winter'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `winter`.
    """
    # normalize the confusion matrix
    os.makedirs(save_dir, exist_ok=True)
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix = \
        confusion_matrix.astype(np.float32) / per_label_sums * 100

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(2 * num_classes, 2 * num_classes * 0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confusion matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}%'.format(
                    round(confusion_matrix[i, j], 2
                          ) if not np.isnan(confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='w',
                size=7)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, name), format='png')
    if show:
        plt.show()

def calculate_metric_from_confusion_matrix(hist):
    acc = np.diag(hist) / hist.sum(axis=1)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    fp = hist[0][1:].sum()/hist[0].sum()
    return iou, acc, fp

    


if __name__ == "__main__":
    # gt_path = "/root/autodl-pub/CZX/mmsegmentation_czx/data/snow_ice_data/label/val"
    gt_path = "/aipt/CZX/mmsegmentation_czx/data/snow_ice_data/label/val"
    pred_path = "/aipt/CZX/mmsegmentation_czx/work_dirs/ice_snow/pred/pred_fcn"
    confusion_m = calculate_confusion_matrix(gt_path, pred_path, 3)
    iou, acc, false_positive_rate = calculate_metric_from_confusion_matrix(confusion_m)
    print("iou",iou)
    print("miou", iou.mean())
    print('acc', acc)
    print("macc", acc.mean())
    print("虚警率", false_positive_rate*100)
    plot_confusion_matrix(confusion_matrix=confusion_m, name='fcn.png')
    
    