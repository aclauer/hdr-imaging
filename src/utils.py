import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def record_color_check_locations():
    image_path = Path("data/door_stack") / Path("exposure14.jpg")
    image = cv.imread(image_path)
    plt.imshow(image)

    points = plt.ginput(48, timeout=-1)

    plt.close()
    # plt.show()

    points = np.array(points, dtype=np.int32)
    print(points)
    np.save("color_locations.npy", points)

if __name__ == "__main__":
    record_color_check_locations()