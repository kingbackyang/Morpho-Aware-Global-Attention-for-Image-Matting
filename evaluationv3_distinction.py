import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import shutil
import scipy.ndimage
from skimage.measure import label
import scipy.ndimage.morphology
import eval_distinction as eval_all

def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.0
    # loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)

    # # if test on whole image (Disitinctions-646), please uncomment this line
    loss = np.sum(error_map ** 2) / (pred.shape[0] * pred.shape[1])

    return loss


def compute_sad_loss(pred, target, trimap):
    error_map = np.abs((pred - target) / 255.0)
    # loss = np.sum(error_map * (trimap == 128))

    # # if test on whole image (Disitinctions-646), please uncomment this line
    loss = np.sum(error_map)

    return loss / 1000, np.sum(trimap == 128) / 1000

def evaluate(args):
    img_names = []
    mse_loss_unknown = []
    sad_loss_unknown = []
    grad_loss_unknown = []
    conn_loss_unknown = []

    mse = eval_all.MSE()
    sad = eval_all.SAD()
    grad = eval_all.Grad()
    conn = eval_all.Conn()


    bad_case = []

    for i, img in tqdm(enumerate(os.listdir(args.label_dir))):

        if not((os.path.isfile(os.path.join(args.pred_dir, img)) and
                os.path.isfile(os.path.join(args.label_dir, img)) and
                os.path.isfile(os.path.join(args.trimap_dir, img)))):
            print('[{}/{}] "{}" skipping'.format(i, len(os.listdir(args.label_dir)), img))
            continue

        pred = cv2.imread(os.path.join(args.pred_dir, img), 0).astype(np.float32)
        label = cv2.imread(os.path.join(args.label_dir, img), 0).astype(np.float32)
        trimap = cv2.imread(os.path.join(args.trimap_dir, img), 0).astype(np.float32)

        mse.update(pred, label, trimap=trimap) # pred [0., 255.]. gt [0, 255].trimap  {0, 128, 255}.
        sad.update(pred, label, trimap=trimap) # pred [0., 255.]. gt [0, 255].trimap  {0, 128, 255}.
        grad.update(pred, label, trimap=trimap) # pred [0., 1.]. gt [0, 255].trimap  {0, 128, 255}.
        conn.update(pred, label, trimap=trimap) # pred [0., 1.]. gt [0, 255].trimap  {0, 128, 255}.

        # calculate loss
        mse_loss_unknown_ = compute_mse_loss(pred, label, trimap)
        sad_loss_unknown_ = compute_sad_loss(pred, label, trimap)[0]


        # save for average
        img_names.append(img)

        mse_loss_unknown.append(mse_loss_unknown_)  # mean l2 loss per unknown pixel
        sad_loss_unknown.append(sad_loss_unknown_)  # l1 loss on unknown area

    print('* Unknown Region: MSE:', np.array(mse_loss_unknown).mean(), ' SAD:', np.array(sad_loss_unknown).mean())
    print('MSE:', mse.evaluate(), 'SAD:', sad.evaluate(), 'Grad:', grad.evaluate(), 'Connectivity:', conn.evaluate())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str, required=True, help="output dir")
    parser.add_argument('--label-dir', type=str, default='', help="GT alpha dir")
    parser.add_argument('--trimap-dir', type=str, default='', help="trimap dir")

    args = parser.parse_args()

    evaluate(args)