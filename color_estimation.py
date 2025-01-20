import cv2
import numpy as np

def detect_cone_color(image, x, y, radius=20, chunks=10):
    # Convert coordinates to integers
    x, y = int(x), int(y)
    height, width = image.shape[:2]
    
    # Define sampling region with bounds checking
    x_min = max(0, x - radius)
    x_max = min(width, x + radius)
    y_min = max(0, y - radius)
    y_max = min(height, y + radius)
    
    # Extract region of interest
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return "unknown", 0.0
    
    # Convert to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    yellow_range = (np.array([20, 100, 100]), np.array([30, 255, 255]))
    blue_range = (np.array([100, 100, 100]), np.array([120, 255, 255]))
    
    # Create color masks
    yellow_mask = cv2.inRange(hsv_roi, yellow_range[0], yellow_range[1])
    blue_mask = cv2.inRange(hsv_roi, blue_range[0], blue_range[1])
    
    # Calculate percentage of pixels matching each color
    total_pixels = (y_max - y_min) * (x_max - x_min)
    yellow_pixels = np.count_nonzero(yellow_mask)
    blue_pixels = np.count_nonzero(blue_mask)
    
    yellow_percentage = yellow_pixels / total_pixels
    blue_percentage = blue_pixels / total_pixels
    
    # Minimum confidence threshold
    MIN_CONFIDENCE = 0.05
    
    # Determine cone color based on percentages
    if yellow_percentage > MIN_CONFIDENCE and yellow_percentage > blue_percentage * 2:
        return "Yellow", yellow_percentage
    elif blue_percentage > MIN_CONFIDENCE and blue_percentage > yellow_percentage * 2:
        return "Blue", blue_percentage
    else:
        return "unknown", max(yellow_percentage, blue_percentage)