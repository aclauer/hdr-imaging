import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from PIL import Image
from PIL.ExifTags import TAGS
from cp_hw2 import writeHDR

def linearize_images(stack_path):
    downsample = 200

    data_path = Path(stack_path)
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

    weighting_scheme = "tent"

    row = 0
    for i in range(1, 254 + 1):
        l = 1
        w = weight(i / 255.0, weighting_scheme) * l
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
                w = weight(channel_intensity / 255.0, weighting_scheme)
                A[row, channel_intensity] = w
                A[row, num_intensities + 3 * i + c] = -w
                b[row] = w * log_t
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

        plt.imshow(image)
        plt.title("Linearized image")
        plt.show()

def weight(z, t_k=None, scheme="uniform"):
    # Z_min, Z_max = 0.000001, 0.9999999
    Z_min, Z_max = 0, 1

    match scheme:
        case "uniform":
            if Z_min <= z and z <= Z_max:
                return 1
            return 0
        case "tent":
            if Z_min <= z and z <= Z_max:
                return min(z, 1-z)
            return 0
        case "gaussian":
            if Z_min <= z and z <= Z_max:
                return np.exp(-4 * (z - 0.5)**2 / 0.5**2)
            return 0
        case "photon":
            if Z_min <= z and z <= Z_max:
                return t_k
            return 0

def load_and_reshape_image(image_path, downsample=1):
    """Reshapes the image so each row is a RGB pixel value"""
    image = cv.imread(image_path)[::downsample, ::downsample]
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

def HDR_linear(stack_path, file_type):
    data_path = Path(stack_path)
    downsample = 20
    
    weighting_scheme = "photon"

    if file_type == "tiff":
        image_path = data_path / Path(f"exposure1.{file_type}")
        print(image_path)
        image = cv.imread(image_path)
    elif file_type == "jpg":
        pass

    HDR_numer = np.zeros(image[::downsample, ::downsample].shape)
    print(f"Numer shape = {HDR_numer.shape}")
    HDR_denom = np.zeros_like(HDR_numer)

    def process_numer_pixel(I_ldr, I_lin, t_k):
        w = weight(I_ldr, t_k, weighting_scheme)
        val = w * I_lin * t_k
        return val

    def process_denom_pixel(I_ldr, t_k):
        val = weight(I_ldr, t_k, weighting_scheme)
        return val

    vector_process_numer_pixel = np.vectorize(process_numer_pixel)
    vector_process_denom_pixel = np.vectorize(process_denom_pixel)
    
    # for each image...
    # assume .tiff for now...
    for img in range(1, 16+1):
        image_path_ldr = stack_path / Path(f"exposure{img}.tiff")
        image_path_lin = stack_path / Path(f"exposure{img}.tiff")

        image_ldr = cv.imread(image_path_ldr, cv.IMREAD_UNCHANGED)[0::downsample, 0::downsample] / (2**16 - 1)
        image_lin = cv.imread(image_path_lin, cv.IMREAD_UNCHANGED)[0::downsample, 0::downsample] / (2**16 - 1)

        # Just use the jpg for the exposure because it's easier
        t_k = get_image_exposure_time(stack_path / Path(f"exposure{img}.jpg"))
        print(f"t_k = {t_k}")

        # mask = np.any(image_ldr < 0.1, axis=2)
        # image_ldr[mask] = 0

        # mask = np.any(image_lin < 0.0, axis=2)
        # image_lin[mask] = 0

        image_numer = vector_process_numer_pixel(image_ldr, image_lin, t_k)
        image_denom = vector_process_denom_pixel(image_ldr, t_k)

        HDR_numer += image_numer
        HDR_denom += image_denom

    HDR_denom[HDR_denom == 0] = 0.01
    HDR_numer[HDR_denom == 0] = 0
    HDR_numer /= np.max(HDR_numer)

    print(HDR_denom.shape)
    print(f"Number of hdr denom zeros = {np.count_nonzero(HDR_denom == 0)}")

    HDR = HDR_numer / HDR_denom
    HDR = HDR / np.mean(HDR) / 2

    # HDR *= 0.5

    # HDR = gamma_encoding(HDR)
    HDR = np.clip(HDR, 0, 1)
    plt.imshow(HDR)
    plt.show()

    writeHDR("test.hdr", HDR)


def gamma_encoding(image):
    def _gamma_encoding(x):
        if x <= 0.0031308:
            return 12.92 * x
        return (1 + 0.055) * (x ** (1 / 2.4)) - 0.055

    vector_gamma = np.vectorize(_gamma_encoding)
    gamma_encoded_image = vector_gamma(image)
    return np.clip(vector_gamma(gamma_encoded_image), 0, 1)


if __name__ == "__main__":
    HDR_linear("data/door_stack", "tiff")
