import os
import cv2
from multiprocessing import Pool

from facedetect import face_detect


if __name__ == "__main__":
    list_image = os.listdir('./images')
    # print (list_image)
    pool = Pool(processes=4)
    output = pool.map_async(face_detect,list_image)

    cv2.imshow('test',(output.get()[1]))
    cv2.waitKey(0)