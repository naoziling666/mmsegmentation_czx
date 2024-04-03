import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import mmengine

def calculate_confusion_matrix(gt_path, pred_path, num_classes, reduce_zero_label=True):
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



def calculate_confusion_matrix_from_pkl(pkl_path, num_classes=4):
    confusion_matrix = np.zeros(shape=[num_classes, num_classes])
    results = mmengine.fileio.load(pkl_path)
    for item in results:
        
        pred = item['pred_sem_seg']['data'].squeeze()
        gt = item['gt_sem_seg']['data'].squeeze()
        gt_segm = np.array(gt).astype(int).reshape(-1)
        res_segm  = np.array(pred).astype(int).reshape(-1)
        k = (gt_segm >= 0) & (gt_segm < num_classes)
        inds = num_classes * gt_segm[k] + res_segm[k]
        inds = inds.flatten()
        mat = np.bincount(inds, minlength=num_classes**2).reshape(num_classes, num_classes)
        confusion_matrix += mat
    iou = np.diag(confusion_matrix) / (confusion_matrix.sum(1) + confusion_matrix.sum(0) - np.diag(confusion_matrix))
    # 计算miou
    miou = np.mean(iou)
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix,
                          name,
                          labels=['land','fog', 'cloud', 'cloud_fog', 'ocean'],
                          save_dir='/home/ps/CZX/mmsegmentation_czx/work_dirs/infer_600_600/confusion_matrix',
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




if __name__ == "__main__":
    # gt_path = "/home/ps/CZX/mmsegmentation_czx/data/seafog_600/crop_mask/val"
    # pred_path = "/home/ps/CZX/mmsegmentation_czx/work_dirs/segnext_mscan_-l_2xb4-adamw-focal_loss-40k_seafog_3band-600*600_neck_channel_attention/20231113-120633/test"
    # confusion_m = calculate_confusion_matrix(gt_path, pred_path, 4)
    pkl_path = '/home/ps/CZX/mmsegmentation_czx/work_dirs/infer_600_600/pkl/yhdata.pkl'
    # pkl 由test.py导出，test.py中有具体命令
    confusion_m = calculate_confusion_matrix_from_pkl(pkl_path, num_classes=2)
    plot_confusion_matrix(confusion_matrix=confusion_m, name='yhdata.png', labels=['fog','other'])
    
    