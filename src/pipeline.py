import multiprocessing
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from PIL import Image
from PIL.ExifTags import TAGS
from cp_hw2 import writeHDR, read_colorchecker_gm, readHDR

d = 1

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
    # Z_min, Z_max = 0.05, 0.95
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

def merge_hdr(stack_path, method, weighting_scheme, file_type):
    data_path = Path(stack_path)
    downsample = d

    if file_type == "tiff":
        image_path = data_path / Path(f"exposure1.{file_type}")
        image = cv.imread(image_path)
    elif file_type == "jpg":
        pass

    HDR_numer = np.zeros(image[::downsample, ::downsample].shape)
    print(f"Numer shape = {HDR_numer.shape}")
    HDR_denom = np.zeros_like(HDR_numer)

    def process_numer_pixel_linear(I_ldr, I_lin, t_k):
        w = weight(I_ldr, t_k, weighting_scheme)
        val = w * I_lin * t_k
        return val

    def process_denom_pixel_linear(I_ldr, t_k):
        val = weight(I_ldr, t_k, weighting_scheme)
        return val

    vector_process_numer_pixel_linear = np.vectorize(process_numer_pixel_linear)
    vector_process_denom_pixel_linear = np.vectorize(process_denom_pixel_linear)
    
    e = 0.000001

    def process_numer_pixel_exponential(I_ldr, I_lin, t_k):
        w = weight(I_ldr, t_k, weighting_scheme)
        val = w * (np.log(I_lin + e) - np.log(t_k))
        return val

    def process_denom_pixel_exponential(I_ldr, t_k):
        val = weight(I_ldr, t_k, weighting_scheme)
        return val

    vector_process_numer_pixel_exponential = np.vectorize(process_numer_pixel_exponential)
    vector_process_denom_pixel_exponential = np.vectorize(process_denom_pixel_exponential)

    # for each image...
    # assume .tiff for now...
    for img in range(1, 16+1):
        # TODO: Path for lin will have to change when working with jpg
        image_path_ldr = stack_path / Path(f"exposure{img}.tiff")
        image_path_lin = stack_path / Path(f"exposure{img}.tiff")

        image_ldr = cv.imread(image_path_ldr, cv.IMREAD_UNCHANGED)[0::downsample, 0::downsample] / (2**16 - 1)
        image_lin = cv.imread(image_path_lin, cv.IMREAD_UNCHANGED)[0::downsample, 0::downsample] / (2**16 - 1)

        # Just use the jpg for the exposure because it's easier
        t_k = get_image_exposure_time(stack_path / Path(f"exposure{img}.jpg"))
        print(f"{method} {weighting_scheme} {file_type}: t_k = {t_k}")

        if method == "linear":
            image_numer = vector_process_numer_pixel_linear(image_ldr, image_lin, t_k)
            image_denom = vector_process_denom_pixel_linear(image_ldr, t_k)

        if method == "exponential":
            image_numer = vector_process_numer_pixel_exponential(image_ldr, image_lin, t_k)
            image_denom = vector_process_denom_pixel_exponential(image_ldr, t_k) 

        HDR_numer += image_numer
        HDR_denom += image_denom

    HDR_denom[HDR_denom == 0] = 0.01
    HDR_numer[HDR_denom == 0] = 0
    HDR_numer /= np.max(HDR_numer)

    print(HDR_denom.shape)
    print(f"Number of hdr denom zeros = {np.count_nonzero(HDR_denom == 0)}")

    HDR = HDR_numer / HDR_denom
    HDR = HDR / np.mean(HDR) / 2

    # if method == "exponential" and weighting_scheme == "uniform":
    # if method == "exponential" and weighting_scheme != "tent" and weighting_scheme != "gaussian":
    if method == "exponential":
        HDR *= 0.75
        HDR = np.clip(HDR, 0, 1)
        HDR = 1 - HDR
    else:
        HDR = gamma_encoding(HDR)
        HDR = np.clip(HDR, 0, 1)

    print(HDR.shape)

    if HDR.dtype == np.float64:
        HDR = HDR.astype(np.float32)

    HDR = cv.cvtColor(HDR, cv.COLOR_BGR2RGB)

    writeHDR(f"merged_images/{method}_{weighting_scheme}_{file_type}.hdr", HDR)
    plt.imsave(f"merged_images/{method}_{weighting_scheme}_{file_type}.png", HDR)


def gamma_encoding(image):
    def _gamma_encoding(x):
        if x <= 0.0031308:
            return 12.92 * x
        return (1 + 0.055) * (x ** (1 / 2.4)) - 0.055

    vector_gamma = np.vectorize(_gamma_encoding)
    gamma_encoded_image = vector_gamma(image)
    return np.clip(vector_gamma(gamma_encoded_image), 0, 1)


def color_correction(hdr_image):
    color_locations = np.load("color_locations.npy")
    rgb_averages = []
    for i in range(0, 48, 2):
        y1, x1 = color_locations[i] / d        # REMOVE IF THERE IS NO DOWN SAMPLING
        y2, x2 = color_locations[i+1] / d
        square = hdr_image[int(x1):int(x2), int(y1):int(y2)]

        averages = square.mean(axis=(0, 1))        
        rgb_averages.append((averages[2], averages[1], averages[0]))

    rgb_averages = np.array(rgb_averages, dtype=np.float32)
    r_gt, g_gt, b_gt = read_colorchecker_gm()

    ground_truth = np.stack([r_gt, g_gt, b_gt], axis=-1).reshape(-1, 3)
    rgb_averages_h = np.hstack([rgb_averages, np.ones((24, 1))])

    A, _, _, _ = np.linalg.lstsq(rgb_averages_h, ground_truth, rcond=None)

    A = A.T

    H, W, _ = hdr_image.shape
    pixels = hdr_image.reshape(-1, 3)

    pixels_h = np.hstack([pixels, np.ones((pixels.shape[0], 1))])
    color_corrected = (A @ pixels_h.T).T

    color_corrected = np.clip(color_corrected, 0, 1)
    corrected_image = color_corrected.reshape(H, W, 3)

    # if corrected_image.dtype == np.float64:
    #     corrected_image = corrected_image.astype(np.float32)
    # corrected_image = cv.cvtColor(corrected_image, cv.COLOR_BGR2RGB)

    # return corrected_image
    
    w1, w2 = color_locations[18], color_locations[19]
    wy1, wx1 = w1 / d # REMOVE IF THERE IS NO DOWNSAMPLING
    wy2, wx2 = w2 / d
    
    image_white_rgb = np.mean(corrected_image[int(wy1):int(wy2), int(wx1):int(wx2)], axis=(0, 1))
    image_white_rgb[image_white_rgb == 0] = 1

    ground_truth_white_rgb = gamma_encoding(np.clip(ground_truth[18], 0, 1))

    scaling_factors = ground_truth_white_rgb / image_white_rgb

    balanced_image = np.zeros_like(hdr_image)

    balanced_image[:, :, 0] = hdr_image[:, :, 0] * scaling_factors[0]
    balanced_image[:, :, 1] = hdr_image[:, :, 1] * scaling_factors[1]
    balanced_image[:, :, 2] = hdr_image[:, :, 2] * scaling_factors[2]

    balanced_image = np.clip(balanced_image / 255.0, 0, 1)
    return balanced_image


def run_full_pipeline(method, scheme):
    print(f"============ Running pipeline for {method} {scheme} ============")
    merge_hdr("data/door_stack", method, scheme, "tiff")

    path = Path(f"merged_images/{method}_{scheme}_tiff.png")
    print(f"Color correcting {path}")
    hdr_image = cv.imread(f"merged_images/{method}_{scheme}_tiff.png")

    corrected_image = color_correction(hdr_image)

    if corrected_image.dtype == np.float64:
        corrected_image = corrected_image.astype(np.float32)
    corrected_image = cv.cvtColor(corrected_image, cv.COLOR_BGR2RGB)

    writeHDR(f"corrected_images/{method}_{scheme}_tiff.hdr", corrected_image)
    plt.imsave(f"corrected_images/{method}_{scheme}_tiff.png", corrected_image)

if __name__ == "__main__":
    # merge_hdr("data/door_stack", "exponential", "tent", "tiff")
    # merge_hdr("data/door_stack", "exponential", "gaussian", "tiff")

    methods = ["linear", "exponential"]
    schemes = ["uniform", "tent", "gaussian", "photon"]

    with multiprocessing.Pool(processes=4) as pool:
        pool.starmap(run_full_pipeline, [(method, scheme) for method in methods for scheme in schemes])

    # run_full_pipeline("linear", "tent")
