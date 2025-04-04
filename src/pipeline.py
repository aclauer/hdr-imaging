import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from PIL import Image
from PIL.ExifTags import TAGS

def main():
    linearize_images()

def linearize_images():
    downsample = 200

    data_path = Path("data/door_stack")
    image_path = data_path / Path(f"exposure1.jpg")

    image = load_and_reshape_image(image_path, downsample=downsample)

    num_images = 16
    num_samples = image.shape[0]
    num_intensities = 256

    num_equations = num_intensities - 2 + num_samples * num_images * 3

    print(f"Estimating g with {num_images} images, {num_samples} samples per image, {num_intensities} intensities ({num_equations} equations)")

    A = np.zeros((num_equations, num_intensities + 3 * num_samples))
    b = np.zeros(num_equations)

    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")

    row = 0
    for i in range(1, 254 + 1):
        l = 1
        w = weight(i / 255.0) * l
        A[row, i-1] = w
        A[row, i] = -2*w
        A[row, i+1] = w
        row += 1

    for img in range(1, 16+1):
        image_path = data_path / Path(f"exposure{img}.jpg")
        image = load_and_reshape_image(image_path, downsample=downsample)

        exposure_time = get_image_exposure_time(image_path)
        log_t = np.log(exposure_time)

        for i in range(num_samples):
            for c in range(3):
                channel_intensity = image[i, c]
                w = weight(channel_intensity / 255.0)
                A[row, channel_intensity] = w
                A[row, num_intensities + 3 * i + c] = -w
                b[row] = w * log_t          # TODO: Add lambda value here
                row += 1

    print("Solving system of equations")
    x, _, _, _ = np.linalg.lstsq(A, b)
    g = x[:num_intensities]

    plt.plot(g)
    plt.title("g (uniform weighting)")
    plt.show()

    for img in range(1, 16+1, 4):

        image_path = data_path / Path(f"exposure{img}.jpg")
        image = cv.imread(image_path)

        plt.imshow(image)
        plt.title("Raw image")
        plt.show()

        image = np.exp(g[image])

        # min_val = np.min(image)
        # max_val = np.max(image)
        # image = (image - min_val) / (max_val - min_val)

        plt.imshow(image)
        plt.title("Linearized image")
        plt.show()

def weight(z, scheme="uniform"):
    match scheme:
        case "uniform":
            Z_min, Z_max = 0.05, 0.95
            if Z_min <= z and z <= Z_max:
                return 1
            return 0

def load_and_reshape_image(image_path, downsample=1):
    """Reshapes the image so each row is a RGB pixel value"""
    # TODO: Double check that it is RGB
    image = cv.imread(image_path)[0::downsample, 0::downsample]
    return image.transpose(2,0,1).reshape(3,-1).transpose()

def get_image_exposure_time(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()
    
    if exif_data is not None:
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == "ExposureTime":
                return float(value)
        
    raise ValueError(f"Exposure time not found in {image_path}")

if __name__ == "__main__":
    main()