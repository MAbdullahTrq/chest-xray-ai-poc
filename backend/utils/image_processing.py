"""
Image processing utilities for chest X-ray analysis
"""

import numpy as np
from PIL import Image
import cv2
import pydicom
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image: Union[Image.Image, np.ndarray], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image for chest X-ray model input
    
    Args:
        image: PIL Image or numpy array
        target_size: Target size for resizing (width, height)
        
    Returns:
        Preprocessed numpy array
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image.copy()
        
        # Handle different image formats
        if len(image_array.shape) == 3:
            # Color image - convert to grayscale
            if image_array.shape[2] == 3:  # RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            elif image_array.shape[2] == 4:  # RGBA
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
        
        # Ensure we have a 2D array
        if len(image_array.shape) > 2:
            image_array = image_array[:, :, 0]
        
        # Resize image
        image_array = cv2.resize(image_array, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [0, 255] range
        image_array = normalize_pixel_values(image_array)
        
        # Apply histogram equalization to improve contrast
        image_array = apply_clahe(image_array)
        
        logger.debug(f"Image preprocessed: shape={image_array.shape}, dtype={image_array.dtype}, range=[{image_array.min()}, {image_array.max()}]")
        
        return image_array
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise e

def normalize_pixel_values(image_array: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 255] range
    
    Args:
        image_array: Input image array
        
    Returns:
        Normalized image array
    """
    # Handle different data types
    if image_array.dtype == np.uint8:
        return image_array
    
    # Normalize to [0, 255] range
    min_val = image_array.min()
    max_val = image_array.max()
    
    if max_val > min_val:
        normalized = ((image_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image_array, dtype=np.uint8)
    
    return normalized

def apply_clahe(image_array: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    
    Args:
        image_array: Input grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of the neighborhood for histogram equalization
        
    Returns:
        CLAHE enhanced image
    """
    try:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image_array)
        return enhanced
    except Exception as e:
        logger.warning(f"CLAHE enhancement failed: {str(e)}. Returning original image.")
        return image_array

def load_dicom_image(dicom_path: str) -> np.ndarray:
    """
    Load DICOM image and convert to numpy array
    
    Args:
        dicom_path: Path to DICOM file
        
    Returns:
        Image array
    """
    try:
        # Read DICOM file
        dicom_data = pydicom.dcmread(dicom_path)
        
        # Extract pixel array
        image_array = dicom_data.pixel_array
        
        # Apply rescale slope and intercept if present
        if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
            slope = float(dicom_data.RescaleSlope)
            intercept = float(dicom_data.RescaleIntercept)
            image_array = image_array * slope + intercept
        
        # Handle photometric interpretation
        if hasattr(dicom_data, 'PhotometricInterpretation'):
            if dicom_data.PhotometricInterpretation == 'MONOCHROME1':
                # Invert image for MONOCHROME1 (where 0 is white)
                image_array = np.max(image_array) - image_array
        
        return image_array.astype(np.float32)
        
    except Exception as e:
        logger.error(f"DICOM loading failed: {str(e)}")
        raise e

def validate_chest_xray_image(image_array: np.ndarray) -> bool:
    """
    Basic validation to check if image looks like a chest X-ray
    
    Args:
        image_array: Input image array
        
    Returns:
        True if image passes basic validation checks
    """
    try:
        # Check image dimensions
        height, width = image_array.shape[:2]
        
        # Chest X-rays are typically taller than they are wide
        aspect_ratio = height / width
        if aspect_ratio < 0.8 or aspect_ratio > 2.0:
            logger.warning(f"Unusual aspect ratio for chest X-ray: {aspect_ratio:.2f}")
            return False
        
        # Check if image has reasonable contrast
        std_dev = np.std(image_array)
        if std_dev < 10:  # Very low contrast
            logger.warning(f"Low contrast image detected: std={std_dev:.2f}")
            return False
        
        # Check for completely black or white images
        mean_intensity = np.mean(image_array)
        if mean_intensity < 5 or mean_intensity > 250:
            logger.warning(f"Unusual intensity distribution: mean={mean_intensity:.2f}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        return False

def detect_image_orientation(image_array: np.ndarray) -> str:
    """
    Detect if chest X-ray is in correct orientation (PA/AP view)
    
    Args:
        image_array: Input image array
        
    Returns:
        Orientation description
    """
    try:
        height, width = image_array.shape[:2]
        
        # Simple heuristic: chest X-rays are typically portrait orientation
        if height > width:
            return "Portrait (likely correct orientation)"
        else:
            return "Landscape (may need rotation)"
            
    except Exception as e:
        logger.error(f"Orientation detection failed: {str(e)}")
        return "Unknown orientation"

def enhance_xray_contrast(image_array: np.ndarray) -> np.ndarray:
    """
    Apply specific contrast enhancement for chest X-rays
    
    Args:
        image_array: Input image array
        
    Returns:
        Enhanced image array
    """
    try:
        # Apply CLAHE
        enhanced = apply_clahe(image_array, clip_limit=3.0, tile_grid_size=(8, 8))
        
        # Apply mild Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Sharpen edges slightly
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel * 0.1)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
        
    except Exception as e:
        logger.warning(f"Contrast enhancement failed: {str(e)}. Returning original image.")
        return image_array
