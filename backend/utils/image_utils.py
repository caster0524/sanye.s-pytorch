"""
Image utility functions
"""
import base64
import io
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()


def base64_to_image(base64_str: str) -> Optional[Image.Image]:
    """Convert base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None


def get_image_info(image: Image.Image) -> Dict[str, Any]:
    """Get basic information about an image"""
    return {
        "width": image.width,
        "height": image.height,
        "format": image.format or "UNKNOWN",
        "mode": image.mode,
        "size_mb": (image.width * image.height * len(image.getbands())) / (1024 * 1024)
    }


def resize_image(
    image: Image.Image,
    target_size: Tuple[int, int],
    keep_aspect: bool = True
) -> Image.Image:
    """Resize image to target size"""
    if keep_aspect:
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize(target_size, Image.Resampling.LANCZOS)


def center_crop(image: Image.Image, crop_size: Tuple[int, int]) -> Image.Image:
    """Center crop an image"""
    width, height = image.size
    crop_width, crop_height = crop_size
    
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    
    return image.crop((left, top, right, bottom))


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """Normalize image with mean and std"""
    image = image.astype(np.float32) / 255.0
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    return (image - mean) / std


def preprocess_for_classification(
    image: Image.Image,
    size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Preprocess image for classification models"""
    # Resize
    image = resize_image(image, size)
    
    # Center crop
    image = center_crop(image, size)
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert to numpy and normalize
    img_array = np.array(image)
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_array = img_array.astype(np.float32) / 255.0
    img_array = (img_array - mean) / std
    
    # Convert to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    return img_array


def create_thumbnail(
    image: Image.Image,
    max_size: Tuple[int, int] = (256, 256)
) -> Image.Image:
    """Create a thumbnail of the image"""
    thumb = image.copy()
    thumb.thumbnail(max_size, Image.Resampling.LANCZOS)
    return thumb


def apply_mask(
    image: Image.Image,
    mask: Image.Image,
    alpha: float = 0.5
) -> Image.Image:
    """Apply a segmentation mask to an image"""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    if mask.mode != "L":
        mask = mask.convert("L")
    
    # Resize mask to match image
    mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    
    # Create colored mask
    colored_mask = Image.merge("RGBA", [
        mask,
        Image.new("L", mask.size, 255),
        Image.new("L", mask.size, 100),
        Image.new("L", mask.size, int(255 * alpha))
    ])
    
    return Image.alpha_composite(image, colored_mask)


def draw_bounding_boxes(
    image: Image.Image,
    boxes: list,
    labels: list = None,
    colors: list = None
) -> Image.Image:
    """Draw bounding boxes on image"""
    if CV2_AVAILABLE:
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            color = colors[i] if colors else (0, 255, 0)
            cv2.rectangle(img_array, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            if labels:
                cv2.putText(img_array, labels[i], (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_array)
    else:
        return image
