import cv2
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.special import gamma
from skimage.measure import label


class MSE:
    """
    Only calculate the unknown region if trimap provided.
    """

    def __init__(self):
        self.mse_diffs = 0
        self.count = 0

    def update(self, pred, gt, trimap=None):
        """
        update metric.
        Args:
            pred (np.ndarray): The value range is [0., 255.].
            gt (np.ndarray): The value range is [0, 255].
            trimap (np.ndarray, optional) The value is in {0, 128, 255}. Default: None.
        """
        if trimap is None:
            trimap = np.ones_like(gt) * 128
        if not (pred.shape == gt.shape == trimap.shape):
            raise ValueError(
                'The shape of `pred`, `gt` and `trimap` should be equal. '
                'but they are {}, {} and {}'.format(pred.shape, gt.shape,
                                                    trimap.shape))
        # pred[trimap == 0] = 0
        # pred[trimap == 255] = 255
        #
        # mask = trimap == 128
        # pixels = float(mask.sum())
        # pred = pred / 255.
        # gt = gt / 255.
        # diff = (pred - gt) * mask
        # mse_diff = (diff**2).sum() / pixels if pixels > 0 else 0


        # distinction 646

        error_map = (pred - gt) / 255.0
        # loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)

        # # if test on whole image (Disitinctions-646), please uncomment this line
        mse_diff = np.sum(error_map ** 2) / (pred.shape[0] * pred.shape[1])

        self.mse_diffs += mse_diff
        self.count += 1

        return mse_diff

    def evaluate(self):
        mse = self.mse_diffs / self.count if self.count > 0 else 0
        return mse


class SAD:
    """
    Only calculate the unknown region if trimap provided.
    """

    def __init__(self):
        self.sad_diffs = 0
        self.count = 0

    def update(self, pred, gt, trimap=None):
        """
        update metric.
        Args:
            pred (np.ndarray): The value range is [0., 255.].
            gt (np.ndarray): The value range is [0., 255.].
            trimap (np.ndarray, optional)L The value is in {0, 128, 255}. Default: None.
        """
        # if trimap is None:
        #     trimap = np.ones_like(gt) * 128
        # if not (pred.shape == gt.shape == trimap.shape):
        #     raise ValueError(
        #         'The shape of `pred`, `gt` and `trimap` should be equal. '
        #         'but they are {}, {} and {}'.format(pred.shape, gt.shape,
        #                                             trimap.shape))
        # pred[trimap == 0] = 0
        # pred[trimap == 255] = 255
        #
        # mask = trimap == 128
        # pred = pred / 255.
        # gt = gt / 255.
        # diff = (pred - gt) * mask
        # sad_diff = (np.abs(diff)).sum()

        # distinction-646
        error_map = np.abs((pred - gt) / 255.0)
        # loss = np.sum(error_map * (trimap == 128))

        # # if test on whole image (Disitinctions-646), please uncomment this line
        sad_diff = np.sum(error_map)



        sad_diff /= 1000
        self.sad_diffs += sad_diff
        self.count += 1

        return sad_diff

    def evaluate(self):
        sad = self.sad_diffs / self.count if self.count > 0 else 0
        return sad


class Grad:
    """
    Only calculate the unknown region if trimap provided.
    Refer to: https://github.com/open-mlab/mmediting/blob/master/mmedit/core/evaluation/metrics.py
    """

    def __init__(self):
        self.grad_diffs = 0
        self.count = 0

    def gaussian(self, x, sigma):
        return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    def dgaussian(self, x, sigma):
        return -x * self.gaussian(x, sigma) / sigma**2

    def gauss_filter(self, sigma, epsilon=1e-2):
        half_size = np.ceil(
            sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = int(2 * half_size + 1)

        # create filter in x axis
        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = self.gaussian(
                    i - half_size, sigma) * self.dgaussian(j - half_size, sigma)

        # normalize filter
        norm = np.sqrt((filter_x**2).sum())
        filter_x = filter_x / norm
        filter_y = np.transpose(filter_x)

        return filter_x, filter_y

    def gauss_gradient(self, img, sigma):
        filter_x, filter_y = self.gauss_filter(sigma)
        img_filtered_x = cv2.filter2D(
            img, -1, filter_x, borderType=cv2.BORDER_REPLICATE)
        img_filtered_y = cv2.filter2D(
            img, -1, filter_y, borderType=cv2.BORDER_REPLICATE)
        return np.sqrt(img_filtered_x**2 + img_filtered_y**2)

    def update(self, pred, gt, trimap=None, sigma=1.4):
        """
        update metric.
        Args:
            pred (np.ndarray): The value range is [0., 1.].
            gt (np.ndarray): The value range is [0, 255].
            trimap (np.ndarray, optional)L The value is in {0, 128, 255}. Default: None.
            sigma (float, optional): Standard deviation of the gaussian kernel. Default: 1.4.
        """
        if trimap is None:
            trimap = np.ones_like(gt) * 128
        if not (pred.shape == gt.shape == trimap.shape):
            raise ValueError(
                'The shape of `pred`, `gt` and `trimap` should be equal. '
                'but they are {}, {} and {}'.format(pred.shape, gt.shape,
                                                    trimap.shape))
        # pred[trimap == 0] = 0
        # pred[trimap == 255] = 255
        #
        # gt = gt.squeeze()
        # pred = pred.squeeze()
        # gt = gt.astype(np.float64)
        # pred = pred.astype(np.float64)
        # gt_normed = np.zeros_like(gt)
        # pred_normed = np.zeros_like(pred)
        # cv2.normalize(gt, gt_normed, 1., 0., cv2.NORM_MINMAX)
        # cv2.normalize(pred, pred_normed, 1., 0., cv2.NORM_MINMAX)
        #
        # gt_grad = self.gauss_gradient(gt_normed, sigma).astype(np.float32)
        # pred_grad = self.gauss_gradient(pred_normed, sigma).astype(np.float32)
        #
        # grad_diff = ((gt_grad - pred_grad)**2 * (trimap == 128)).sum()

        # distinction-646

        gt = gt.squeeze()
        pred = pred.squeeze()
        gt = gt.astype(np.float64)
        pred = pred.astype(np.float64)
        gt_normed = np.zeros_like(gt)
        pred_normed = np.zeros_like(pred)
        cv2.normalize(gt, gt_normed, 1., 0., cv2.NORM_MINMAX)
        cv2.normalize(pred, pred_normed, 1., 0., cv2.NORM_MINMAX)

        gt_grad = self.gauss_gradient(gt_normed, sigma).astype(np.float32)
        pred_grad = self.gauss_gradient(pred_normed, sigma).astype(np.float32)

        grad_diff = ((gt_grad - pred_grad) ** 2 ).sum()

        grad_diff /= 1000
        self.grad_diffs += grad_diff
        self.count += 1

        return grad_diff

    def evaluate(self):
        grad = self.grad_diffs / self.count if self.count > 0 else 0
        return grad


class Conn:
    """
    Only calculate the unknown region if trimap provided.
    Refer to: Refer to: https://github.com/open-mlab/mmediting/blob/master/mmedit/core/evaluation/metrics.py
    """

    def __init__(self):
        self.conn_diffs = 0
        self.count = 0

    def update(self, pred, gt, trimap=None, step=0.1):
        """
        update metric.
        Args:
            pred (np.ndarray): The value range is [0., 1.].
            gt (np.ndarray): The value range is [0, 255].
            trimap (np.ndarray, optional)L The value is in {0, 128, 255}. Default: None.
            step (float, optional): Step of threshold when computing intersection between
            `gt` and `pred`. Default: 0.1.
        """
        if trimap is None:
            trimap = np.ones_like(gt) * 128
        if not (pred.shape == gt.shape == trimap.shape):
            raise ValueError(
                'The shape of `pred`, `gt` and `trimap` should be equal. '
                'but they are {}, {} and {}'.format(pred.shape, gt.shape,
                                                    trimap.shape))
        # pred[trimap == 0] = 0
        # pred[trimap == 255] = 255
        #
        # gt = gt.squeeze()
        # pred = pred.squeeze()
        # gt = gt.astype(np.float32) / 255
        # pred = pred.astype(np.float32) / 255
        #
        # thresh_steps = np.arange(0, 1 + step, step)
        # round_down_map = -np.ones_like(gt)
        # for i in range(1, len(thresh_steps)):
        #     gt_thresh = gt >= thresh_steps[i]
        #     pred_thresh = pred >= thresh_steps[i]
        #     intersection = (gt_thresh & pred_thresh).astype(np.uint8)
        #
        #     # connected components
        #     _, output, stats, _ = cv2.connectedComponentsWithStats(
        #         intersection, connectivity=4)
        #     # start from 1 in dim 0 to exclude background
        #     size = stats[1:, -1]
        #
        #     # largest connected component of the intersection
        #     omega = np.zeros_like(gt)
        #     if len(size) != 0:
        #         max_id = np.argmax(size)
        #         # plus one to include background
        #         omega[output == max_id + 1] = 1
        #
        #     mask = (round_down_map == -1) & (omega == 0)
        #     round_down_map[mask] = thresh_steps[i - 1]
        # round_down_map[round_down_map == -1] = 1
        #
        # gt_diff = gt - round_down_map
        # pred_diff = pred - round_down_map
        # # only calculate difference larger than or equal to 0.15
        # gt_phi = 1 - gt_diff * (gt_diff >= 0.15)
        # pred_phi = 1 - pred_diff * (pred_diff >= 0.15)
        #
        # conn_diff = np.sum(np.abs(gt_phi - pred_phi) * (trimap == 128))

        # distinction-646
        gt = gt.squeeze()
        pred = pred.squeeze()
        gt = gt.astype(np.float32) / 255
        pred = pred.astype(np.float32) / 255

        thresh_steps = np.arange(0, 1 + step, step)
        round_down_map = -np.ones_like(gt)
        for i in range(1, len(thresh_steps)):
            gt_thresh = gt >= thresh_steps[i]
            pred_thresh = pred >= thresh_steps[i]
            intersection = (gt_thresh & pred_thresh).astype(np.uint8)

            # connected components
            _, output, stats, _ = cv2.connectedComponentsWithStats(
                intersection, connectivity=4)
            # start from 1 in dim 0 to exclude background
            size = stats[1:, -1]

            # largest connected component of the intersection
            omega = np.zeros_like(gt)
            if len(size) != 0:
                max_id = np.argmax(size)
                # plus one to include background
                omega[output == max_id + 1] = 1

            mask = (round_down_map == -1) & (omega == 0)
            round_down_map[mask] = thresh_steps[i - 1]
        round_down_map[round_down_map == -1] = 1

        gt_diff = gt - round_down_map
        pred_diff = pred - round_down_map
        # only calculate difference larger than or equal to 0.15
        gt_phi = 1 - gt_diff * (gt_diff >= 0.15)
        pred_phi = 1 - pred_diff * (pred_diff >= 0.15)

        conn_diff = np.sum(np.abs(gt_phi - pred_phi))

        conn_diff /= 1000
        self.conn_diffs += conn_diff
        self.count += 1

        return conn_diff

    def evaluate(self):
        conn = self.conn_diffs / self.count if self.count > 0 else 0
        return conn