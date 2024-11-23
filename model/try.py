import cv2
import os

# Initialize the camera
camera = cv2.VideoCapture(0)

# Define a function to capture an image
def capture_image(phobia_name, save_path):
    print(f"Showing stimuli for {phobia_name}... Look at the camera!")
    input("Press Enter when ready to capture...")
    
    ret, frame = camera.read()
    if ret:
        # Save the captured image
        img_path = os.path.join(save_path, f"{phobia_name}_expression.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Image saved at {img_path}")
    else:
        print("Failed to capture image")

# Example usage
capture_image("arachnophobia", "./captured_images")
