import cv2
import numpy as np
from color_estimation import detect_cone_color

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale coordinates back to original image size
        original_image, display_image = param
        scale_x = original_image.shape[1] / display_image.shape[1]
        scale_y = original_image.shape[0] / display_image.shape[0]
        
        original_x = int(x * scale_x)
        original_y = int(y * scale_y)
        
        # Call the color detection function
        color, conf = detect_cone_color(original_image, original_x, original_y)
        print(f"({original_x}, {original_y}): {color} | conf: {conf:.3f}")
        
        # Draw marker on display image
        cv2.circle(display_image, (x, y), 2, (255,255,255), -1)
        text = f"{color.upper()} ({conf:.2f})"
        cv2.putText(display_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

def main():
    # Read the image
    image = cv2.imread("test3.png")
    if image is None:
        return
        
    display_image = cv2.resize(image, (2000, 1000))
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback, param=(image, display_image))
    
    while True:
        cv2.imshow("Image", display_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()