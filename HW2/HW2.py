import cv2
import numpy as np
import random
import math
import sys

img_list = []
img_gray_list = []

# read the image file & output the color & gray image


def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector


def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ", img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main


def creat_im_window(window_name, img):
    cv2.imshow(window_name, img)

# show the all window you call before im_show()
# and press any key to close all windows


def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_all_imgs():
    baseline_path = ['./baseline/m' + str(i) + '.jpg' for i in range(1, 7)]
    bonus_path = ['./bonus/m' + str(i) + '.jpg' for i in range(1, 5)]

    all_baseline_imgs = []
    all_baseline_imgs_gray = []
    all_bonus_imgs = []
    all_bonus_imgs_gray = []

    for path in baseline_path:
        img, img_gray = read_img(path)
        all_baseline_imgs.append(img)
        all_baseline_imgs_gray.append(img_gray)

    for path in bonus_path:
        img, img_gray = read_img(path)
        all_bonus_imgs.append(img)
        all_bonus_imgs_gray.append(img_gray)

    return all_baseline_imgs, all_baseline_imgs_gray, all_bonus_imgs, all_bonus_imgs_gray


def euclidean_distance(des1, des2):
    return float(np.sqrt(np.sum(np.square(des1 - des2))))


def match_keypoints(kp1, des1, kp2, des2, k=2, ratio=0.75):
    matches = []

    # Brute-force kNN
    # Iterate over keypoints in first image
    for i in range(len(des1)):
        best_match, second_best_match = None, None
        best_distance, second_distance = float('inf'), float('inf')

        # Iterate over keypoints in second image
        for j in range(len(des2)):
            # Compute Euclidean distance between descriptors
            distance = euclidean_distance(des1[i], des2[j])

            # Update best and second-best matches if necessary
            if distance < best_distance:
                second_best_match = best_match
                second_distance = best_distance
                best_match = j
                best_distance = distance
            elif distance < second_distance:
                second_best_match = j
                second_distance = distance

        matches.append([(i, best_match, best_distance),
                       (i, second_best_match, second_distance)])

    good_matches = []
    for m, n in matches:
        if m[2] < ratio * n[2]:
            good_matches.append(m[0:2])

    good_pts = []
    for idx in good_matches:
        good_pts.append(list(kp1[idx[0]].pt + kp2[idx[1]].pt))

    return np.array(good_matches), np.array(good_pts)


def get_random_points(matches, good_pts, k=4):
    idx = random.sample(range(len(matches)), k)  # 4個點的index
    S = [good_pts[i] for i in idx]
    return np.array(S)


def find_Homography(S):
    num_points = len(S)
    A = np.zeros((8, 9), np.float64)
    for i in range(num_points):
        # row 1
        A[i*2][0] = S[i][0]
        A[i*2][1] = S[i][1]
        A[i*2][2] = 1

        A[i*2][6] = -1 * S[i][0] * S[i][2]
        A[i*2][7] = -1 * S[i][1] * S[i][2]
        A[i*2][8] = -1 * S[i][2]

        # row 2
        A[i*2+1][3] = S[i][0]
        A[i*2+1][4] = S[i][1]
        A[i*2+1][5] = 1

        A[i*2+1][6] = -1 * S[i][0] * S[i][3]
        A[i*2+1][7] = -1 * S[i][1] * S[i][3]
        A[i*2+1][8] = -1 * S[i][3]

    U, s, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2]  # normalize h33 to 1
    return H


def compute_inliers(good_pts, H, threshold):
    inliers = 0
    num_pts = len(good_pts)
    p1 = np.concatenate((good_pts[:, 0:2], np.ones((num_pts, 1))), axis=1)
    p2 = good_pts[:, 2:4]
    for i in range(num_pts):
        p2_prime = np.dot(H, p1[i])
        p2_prime = (p2_prime/p2_prime[2])[0:2]
        distance = np.linalg.norm(p2[i] - p2_prime)
        if distance < threshold:
            inliers += 1

    return inliers


def ransac(matches, good_pts, threshold=5, iter=2000):
    max_inliers = 0
    best_H = None
    for i in range(iter):
        S = get_random_points(matches, good_pts)

        H = find_Homography(S)

        inliers = compute_inliers(good_pts, H, threshold)
        if inliers > max_inliers:
            max_inliers = inliers
            best_H = H

    return best_H


def warp_image(left, right, H):
    print("warp images ...")
    hl, wl = left.shape[:2]
    hr, wr = right.shape[:2]
    corners1 = np.array([[0, 0, 1], [0, hl, 1], [wl, 0, 1], [wl, hl, 1]])  # left_img 4 corners
    corners_pers = np.dot(H, corners1.T)
    corners_pers = (corners_pers/corners_pers[2])[0:2]  # normalize
    print(corners_pers)
    

    x_prime = min(min(corners_pers[0]), 0)
    y_prime = min(min(corners_pers[1]), 0)
    
    size = (wr + abs(round(x_prime)), hr + abs(round(y_prime)))

    A = np.array([[1, 0, -x_prime], [0, 1, -y_prime],
                 [0, 0, 1]], dtype=np.float64)
    homography = np.dot(A, H)

    warped_l = cv2.warpPerspective(src=left, M=homography, dsize=size)
    warped_r = cv2.warpPerspective(src=right, M=A, dsize=size)

    return warped_l, warped_r


def blending(left, right):
    blended = np.zeros_like(left)
    height, weight = left.shape[:2]

    for i in range(height):
        for j in range(weight):
            pixel_l = left[i, j, :]
            pixel_r = right[i, j, :]

            if np.sum(pixel_l) >= np.sum(pixel_r):
                blended[i, j, :] = pixel_l
            else:
                blended[i, j, :] = pixel_r

    return blended


def stitcher(img1, img1_gray, img2, img2_gray):
    SIFT_Detector = cv2.SIFT_create()

    # SIFT
    kp1, des1 = SIFT_Detector.detectAndCompute(img1_gray, None)
    kp2, des2 = SIFT_Detector.detectAndCompute(img2_gray, None)

    # Feature matching KNN
    good_matches, good_pts = match_keypoints(kp1, des1, kp2, des2)

    # RANSAC and Homography
    H = ransac(good_matches, good_pts)

    # warp image
    warped_l, warped_r = warp_image(img1, img2, H)

    blended = blending(warped_l, warped_r)
    """ 
    # Feature matching debugging
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    for i, match in enumerate(good):
        if (good_matches[i][0] != match.queryIdx) or (good_matches[i][1] != match.trainIdx):
            print('false')

    # Homography debugging
    scr_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    verify_H, mask = cv2.findHomography(scr_pts, dst_pts, cv2.RANSAC, 5.0)
    print(verify_H, H)
    """

    return blended


if __name__ == '__main__':
    baseline_imgs, baseline_imgs_gray, bonus_imgs, bonus_imgs_gray = read_all_imgs()

    # baseline
    stitch_img = stitcher(
        baseline_imgs[0], baseline_imgs_gray[0], baseline_imgs[1], baseline_imgs_gray[1])
    cv2.imwrite("result_baseline_2.jpg", stitch_img)
    for i in range(2, 6):
        stitch_img = stitcher(stitch_img, img_to_gray(
            stitch_img), baseline_imgs[i], baseline_imgs_gray[i])
        cv2.imwrite("result_baseline_" + str(i+1) + ".jpg", stitch_img)

    # bonus
    stitch_img = stitcher(
        bonus_imgs[0], bonus_imgs_gray[0], bonus_imgs[1], bonus_imgs_gray[1])
    cv2.imwrite("result_bonus_2.jpg", stitch_img)
    for i in range(2, 4):
        stitch_img = stitcher(stitch_img, img_to_gray(
            stitch_img), bonus_imgs[i], bonus_imgs_gray[i])
        cv2.imwrite("result_bonus_" + str(i+1) + ".jpg", stitch_img)

    creat_im_window('Result', stitch_img)
    im_show()
