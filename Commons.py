import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

def read_image(filepath):
    # Read image
    img = cv.imread(filepath)
    if img is None:
        raise FileNotFoundError("File not found: %s" % filepath)
    return img


def img_reverse(img):
    return cv.bitwise_not(img)


def img_split(img, mode = "left", ratio = 0.2, fill = False):
    # default: get left 20% of image
    if mode == "left":
        img_split = img[:, :int(img.shape[1]*ratio)]
    elif mode == "right":
        img_split = img[:, int(img.shape[1]*(1-ratio)):]
    elif mode == "top":
        img_split = img[:int(img.shape[0]*ratio), :]
    elif mode == "bottom":
        img_split = img[int(img.shape[0]*(1-ratio)):, :]
    else:
        img_split = img
    if fill:
        # 如果要填充的话，就把裁掉的部分填充为灰色
        filled_img = np.full_like(img, 128)
        if mode == "left":
            filled_img[:, :img_split.shape[1]] = img_split
        elif mode == "right":
            filled_img[:, -img_split.shape[1]:] = img_split
        elif mode == "top":
            filled_img[:img_split.shape[0], :] = img_split
        elif mode == "bottom":
            filled_img[-img_split.shape[0]:, :] = img_split
        else:
            filled_img = img_split
    else:
        filled_img = img_split

    return filled_img


def img_show(img, winname = "img"):
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, img)
    cv.waitKey(0)
    try:
        cv.destroyWindow(winname)
    except cv.error:
        pass


def img_reverse(img):
    return cv.bitwise_not(img)


def get_dominant_color(image, k=2):
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)

    # Perform k-means clustering to find the most dominant colors
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get the RGB values of the cluster centers
    colors = kmeans.cluster_centers_

    # Get the number of pixels in each cluster
    counts = np.bincount(kmeans.labels_)

    # Find the most dominant color
    dominant_color = colors[counts.argmax()]

    return dominant_color


# 获取所有物体
def get_objects(img):
    # 裁切图片
    img = img_split(img, mode = "left", ratio = 0.18)
    img = img_split(img, mode = "bottom", ratio = 0.89)
    img = img_split(img, mode = "right", ratio = 0.7)

    img_drawn = img.copy()

    # 灰度化
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 二值化
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
    # 腐蚀
    img = img_erode(img, kernel_size=3)

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(img, connectivity=8)
    img = img_size_masking(img, stats, labels, 500, 0)

    img = img_reverse(img)

    # 轮廓检测
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    objects = []

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        mask = img[y:y+h, x:x+w]
        object = img_drawn[y:y + h, x:x + w]
        # 为了方便，将背景变为灰色
        object[mask == 0] = 120

        img_show(object)

        objects.append(object)

    return objects


# 膨胀
def img_dilate(img, kernel_size=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv.dilate(img, kernel, iterations = 1)
    return img


# 腐蚀
def img_erode(img, kernel_size=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv.erode(img, kernel, iterations = 1)
    return img


# 剔除面积过小或过大的连通域
def img_size_masking(img, stats, labels, threshold_area = 500, mode = 0):
    assert mode == 1 or mode == 0

    if mode == 0:
        small_areas = np.where(stats[:, cv.CC_STAT_AREA] < threshold_area)[0]
        labels[np.isin(labels, small_areas)] = 0
    elif mode == 1:
        small_areas = np.where(stats[:, cv.CC_STAT_AREA] > threshold_area)[0]
        labels[np.isin(labels, small_areas)] = 0

    result = np.uint8(labels > 0) * 255

    return result


def find_target(object, target_img):
    object = cv.cvtColor(object, cv.COLOR_BGR2GRAY)
    target_img = cv.cvtColor(target_img, cv.COLOR_BGR2GRAY)

    # 匹配模板
    result = cv.matchTemplate(target_img, object, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    return max_loc

