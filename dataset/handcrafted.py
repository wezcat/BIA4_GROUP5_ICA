import numpy as np
import cv2
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.exposure import rescale_intensity

def extract_handcrafted_features_vector(img_rgb: np.ndarray) -> np.ndarray:
    feats = []

    # resize all images
    H, W = 256, 256
    img = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

    # color space：RGB + HSV
    img_float = img.astype(np.float32) / 255.0
    img_hsv = (rgb2hsv(img_float) * 255.0).astype(np.uint8)

    # mean/std of RGB and HSV
    for arr in (img, img_hsv):
        for c in range(3):
            ch = arr[..., c].astype(np.float32)
            feats.append(ch.mean())
            feats.append(ch.std())

    gray = (rgb2gray(img_float) * 255).astype(np.uint8)
    gray_f = gray.astype(np.float32)
    feats.append(gray_f.mean())              # gray_mean
    feats.append(gray_f.std())               # gray_std
    feats.append(float(gray_f.min()))        # gray_min
    feats.append(float(gray_f.max()))        # gray_max

    # histogram of gray
    hist, _ = np.histogram(gray, bins=32, range=(0, 255), density=True)
    hist = hist + 1e-12
    entropy = -np.sum(hist * np.log2(hist))
    feats.append(float(entropy))             # gray_entropy

    # GLCM
    q = np.floor(rescale_intensity(gray, in_range="image", out_range=(0, 7))).astype(np.uint8)
    distances = [1, 2, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(q, distances=distances, angles=angles,
                        levels=8, symmetric=True, normed=True)

    glcm_props = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    for prop in glcm_props:
        vals = graycoprops(glcm, prop).ravel()
        feats.append(float(vals.mean()))
        feats.append(float(vals.std()))

    # LBP
    P, R = 8, 1
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist_lbp, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    feats.extend([float(v) for v in hist_lbp])

    return np.array(feats, dtype=np.float32)