import cv2
import numpy as np
import matplotlib.pyplot as plt


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
            # np.linalg.norm is comparatively faster
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


def stitch_background(img1, img2, savepath=''):
    try:
        row1, col1, _ = img1.shape
        row2, col2, _ = img2.shape
    except:
        print("Images not available. Please mention correct path and try again.")
        return

    print("Finding similar features...")
    kptsA, kptsB = feature_matching(img1, img2)
    homography, _ = cv2.findHomography(kptsA, kptsB, cv2.RANSAC, 4.0)
    # Assuming imags don't overlap
    # if the number of matching keypoints is <100
    if len(kptsA) < 100:
        print("NO OVERLAP in the given images.")
        return

    print("Stitching Images....")
    imCorner2 = np.float32([[[0, 0]],
                            [[0, row2]],
                            [[col2, row2]],
                            [[col2, 0]]])

    imCorner1 = np.float32([[[0, 0]],
                            [[0, row1]],
                            [[col1, row1]],
                            [[col1, 0]]])

    pTransformed = cv2.perspectiveTransform(imCorner1, homography)
    transformedMat = np.concatenate((imCorner2, pTransformed), axis=0)

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
    imgWarped = cv2.warpPerspective(img1, homographyT, (widthMod, heightMod))

    overlap = imgWarped[-Ymin:row2-Ymin, -Xmin:col2-Xmin]
    img2 = np.where(overlap > img2, overlap, img2)
    imgWarped[-Ymin:row2-Ymin, -Xmin:col2-Xmin] = img2
    cv2.imwrite(savepath, imgWarped)
    print("Done.")
    return


if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)
