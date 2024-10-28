import os
import cv2
import numpy as np
import glob



for imgpath in glob.glob("/media/kemove/403plus/yangjingru/Distinctions-646/Final/Test/GT/*.png"):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img_alpha = cv2.imread("/media/kemove/403plus/yangjingru/ViTMatte-main/Composition-1k-testset/trimaps/brandy-402572_1920_13.png")
    # img_alpha = cv2.cvtColor(img_alpha, cv2.COLOR_BGR2GRAY)

    mask1 = np.zeros(img.shape, dtype=np.uint8)
    mask1[img>245] = 255
    mask1[img<=240] = 0

    mask2 = np.zeros(img.shape, dtype=np.uint8)
    mask2[img>0] = 255
    mask2[img<=0] = 0

    trimap = np.zeros(img.shape, dtype=np.uint8)
    trimap[np.where((mask1==0) & (mask2==255))] = 128
    trimap[np.where(mask1==255)] = 255

    trimap_dilate = np.zeros(img.shape, dtype=np.uint8)
    trimap_dilate[np.where(trimap>127)] = 128

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    trimap_dilate = cv2.erode(trimap_dilate, kernel, iterations=3)

    kernel_size = 8
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    trimap_dilate = cv2.dilate(trimap_dilate, kernel, iterations=8)

    trimap_dilate[np.where(trimap==255)] = 255

    # kernel_size = 5
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # trimap_dilate = cv2.erode(trimap_dilate, kernel, iterations=3)
    #
    # kernel_size = 8
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # trimap_dilate = cv2.dilate(trimap_dilate, kernel, iterations=3)

    # cv2.namedWindow("demo1", cv2.WINDOW_NORMAL)
    # cv2.imshow("demo1", img_alpha)
    #
    # cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
    # cv2.imshow("demo", trimap_dilate)
    # cv2.waitKey()
    nimgpath = imgpath.replace("GT", "ALPHA")
    nsubdir = os.path.dirname(nimgpath)
    os.makedirs(nsubdir, exist_ok=True)
    cv2.imwrite(nimgpath, trimap_dilate)