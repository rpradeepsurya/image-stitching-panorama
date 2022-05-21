import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool


def feature_matching(img1, img2):
    '''
    Input: two 3-channel images
    Return: Coordinates of keypoints that match one image
            with the other
    '''
    sift = cv2.xfeatures2d.SIFT_create(
        contrastThreshold=0.04, edgeThreshold=10)
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    matches = []

    for ds1 in range(len(descriptors1)):
        smallest_distance = np.inf
        second_smallest = np.inf

        for ds2 in range(len(descriptors2)):
            distance = np.linalg.norm(descriptors1[ds1] - descriptors2[ds2])
            #distance = np.sqrt(np.sum((descriptors1[ds1] - descriptors2[ds2])**2))

            if distance < smallest_distance:
                second_smallest = smallest_distance
                smallest_distance = distance
                smallest_index = [ds1, ds2]

            elif distance < second_smallest:
                second_smallest = distance

        # ratio test
        if (smallest_distance/second_smallest) < 0.8:
            matches.append(smallest_index)

    kp_coordinates_1 = []
    kp_coordinates_2 = []

    for val in np.array(matches)[:, 0]:
        kp_coordinates_1.append(keypoints1[val].pt)

    for val in np.array(matches)[:, 1]:
        kp_coordinates_2.append(keypoints2[val].pt)

    return np.array(kp_coordinates_1, dtype="float32"), np.array(kp_coordinates_2, dtype="float32")


def warp(image1, image2, homography):
    row1, col1, _ = image1.shape
    row2, col2, _ = image2.shape
    imCorner2 = np.float32([[[0, 0]],
                            [[0, row2]],
                            [[col2, row2]],
                            [[col2, 0]]])
    imCorner1 = np.float32([[[0, 0]],
                            [[0, row1]],
                            [[col1, row1]],
                            [[col1, 0]]])

    pTransformed = cv2.perspectiveTransform(imCorner2, homography)
    transformedMat = np.concatenate((imCorner1, pTransformed), axis=0)

    XYend = np.max(transformedMat, axis=0)[0].astype(int)
    XYorigin = np.min(transformedMat, axis=0)[0].astype(int)

    Xmin = XYorigin[0]
    Xmax = XYend[0]

    Ymin = XYorigin[1]
    Ymax = XYend[1]

    widthMod = Xmax - Xmin
    heightMod = Ymax - Ymin
    homographyT = np.array(
        [[1, 0, -Xmin], [0, 1, -Ymin], [0, 0, 1]]).dot(homography)

    imgWarped = cv2.warpPerspective(
        image2, homographyT, (widthMod, heightMod), flags=cv2.INTER_LINEAR)
    imgWarped[-Ymin:row1-Ymin, -Xmin:col1-Xmin] = image1
    #cv2.imwrite(savepath, imgWarped)
    return imgWarped


def rescale_img(img):
    dim = sum(img.shape[:2])
    # print("---dimension----")
    # print(dim)
    if dim < 1000:
        return img
    else:
        width = img.shape[1]
        height = img.shape[0]
        scale = 0.8
        while dim > 1000:
            width = int(width * scale)
            height = int(height * scale)
            dim = height + width
        #print("--out of dim--")
        return cv2.resize(img, (width, height, img.shape[2]), interpolation=cv2.INTER_NEAREST)



def stitch(imgmark, N=4, savepath=''):

    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1, N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    
    def get_feature_matches(stitch_combinations):
        #print("inside pool feature")
        pA, pB = feature_matching(
            imgs[stitch_combinations[0]], imgs[stitch_combinations[1]])
        return len(pA), (pA, pB)

    img_indexes = [i for i in range(len(imgs))]
    stitch_combinations = list(itertools.combinations(img_indexes, 2))
    print("Finding stitching order based on good matches.....")

    try:
        #pool = Pool()  # Creating a multiprocessing Pool
        print("Ordering in progress.....")
        matches = pool.map(get_feature_matches, stitch_combinations)
        match_points = []
        match_len = []
        for cont in matches:
            match_len.append(cont[0])
            match_points.append(cont[1])
    except:
        print("Multi core processing not supported, switching to single core processing.")
        print()
        match_len = []
        match_points = []
        print("Ordering in progress.....")
        for comb in stitch_combinations:
            #print(comb)
            pA, pB = feature_matching(imgs[comb[0]], imgs[comb[1]])
            match_len.append(len(pA))
            match_points.append([pA, pB])

    # part of stitch iff overlap >= 20% 
    # taking threshold value as the 20% of maximum number of matches 
    overlap_threshold = max(match_len)*0.20
    img_idx_order = np.where(np.array(match_len) >= overlap_threshold)[0]

    overlap_arr = np.identity(N)

    for idx in img_idx_order:
        x = stitch_combinations[idx][0]
        y = stitch_combinations[idx][1]
        overlap_arr[x][y] = 1
        overlap_arr[y][x] = 1

    print("Stitching images.....")

    img_idx_list = {i for i in range(len(imgs))}
    print(img_idx_list)

    # while img_idx_list:
    for idex in img_idx_order:
        print("Images remaining to stitch = ", len(img_idx_list))
        if len(img_idx_list) == 0:
            break

        img_ind1 = stitch_combinations[idex][0]
        img_ind2 = stitch_combinations[idex][1]

        ind1_present = img_ind1 in img_idx_list
        ind2_present = img_ind2 in img_idx_list

        if (ind1_present and ind2_present):
            ptsAB = match_points[idex]
            matches_A, matches_B = ptsAB[0], ptsAB[1]

            if len(matches_A) > overlap_threshold:
                homography, _ = cv2.findHomography(
                    matches_A, matches_B, cv2.RANSAC, 4.0)
                panorama = warp(imgs[img_ind2], imgs[img_ind1], homography)
                img_idx_list.remove(img_ind1)
                img_idx_list.remove(img_ind2)

        elif ind1_present:
            matches_A, matches_B = feature_matching(panorama, imgs[img_ind1])

            if len(matches_A) > overlap_threshold:
                homography, _ = cv2.findHomography(
                    matches_A, matches_B, cv2.RANSAC, 4.0)
                panorama = warp(imgs[img_ind1], panorama, homography)
                img_idx_list.remove(img_ind1)

        elif ind2_present:
            matches_A, matches_B = feature_matching(panorama, imgs[img_ind2])

            if len(matches_A) > overlap_threshold:
                homography, _ = cv2.findHomography(
                    matches_A, matches_B, cv2.RANSAC, 4.0)
                panorama = warp(imgs[img_ind2], panorama, homography)
                img_idx_list.remove(img_ind2)

        else:
            pass

    print("Done.")
    cv2.imwrite(savepath, panorama)
    return overlap_arr.astype(int)


if __name__ == "__main__":
    pool = Pool()
    # task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    # bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
