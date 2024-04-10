import cv2 as cv
from cv2 import aruco

# dictionary to specify type of the marker
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)

# MARKER_ID = 0
MARKER_SIZE_1 = 378  # pixels
MARKER_SIZE_2 = 567
# generating unique IDs using for loop
'''for id in [100, 105, 200, 205, 400, 405]:  # genereting 20 markers
    # using funtion to draw a marker
    marker_image = aruco.generateImageMarker(marker_dict, id, MARKER_SIZE_2)
    #cv.imshow("img", marker_image)
    cv.imwrite(f"markers_2/marker_{id}.png", marker_image)
    # cv.waitKey(0)
    # break'''

for id in [800]:  # genereting 20 markers
    # using funtion to draw a marker
    marker_image = aruco.generateImageMarker(marker_dict, id, MARKER_SIZE_1)
    #cv.imshow("img", marker_image)
    cv.imwrite(f"markers_2/marker_{id}.png", marker_image)
    # cv.waitKey(0)
    # break