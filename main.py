import os
import gc
import sys
import argparse
import cv2
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt

def create_default_image(filename: str = "default_image.png") -> np.ndarray:
    """
    Default image with contours creator.
    """
    print(f"File '{filename}' not found. Creating default image.")
    image: np.ndarray = np.zeros((300, 500, 3), dtype="uint8")
    
    # Draw 10 shapes (5 circles, 5 rectangles)
    for i in range(5):
        cv2.circle(image, (60 + i * 90, 75), 30, (255, 255, 255), -1)
        cv2.rectangle(image, (30 + i * 90, 175), (90 + i * 90, 235), (255, 255, 255), -1)

    cv2.imwrite(filename, image)
    return image

def load_image(image_path: Optional[str]) -> Optional[np.ndarray]:
    """
    Loads an image from the specified path.
    If no path or the file is missing, creates a default image.
    """
    default_filename: str = "default_image.png"

    if image_path:
        if not os.path.exists(image_path):
            print(f"Error: file '{image_path}' not found.")
            return None
        return cv2.imread(image_path)

    if os.path.exists(default_filename):
        print(f"Using default image: '{default_filename}'")
        return cv2.imread(default_filename)
    else:
        return create_default_image(default_filename)

def find_and_draw_contours(image: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Find contours and draw them.
    """
    image_with_contours: np.ndarray = image.copy()

    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    return image_with_contours, len(contours)

def display_or_save_image(original_image: np.ndarray, processed_image: np.ndarray, output_mode: str) -> None:
    """
    Display or save the original and processed images based on the output mode.
    """
    if output_mode == 'cv2':
        combined_image = np.hstack((original_image, processed_image))
        cv2.imshow("Original | Contours", combined_image)
        cv2.waitKey(0)
    elif output_mode == 'plt':
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        axes[0].imshow(original_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(processed_rgb)
        axes[1].set_title("Contours")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()
    elif output_mode == 'file':
        combined_image = np.hstack((original_image, processed_image))
        output_filename = "result.png"
        cv2.imwrite(output_filename, combined_image)
        print(f"Combined image saved to '{output_filename}'")

def main() -> None:
    """
    Init app
    """
    parser = argparse.ArgumentParser(description="Finding contours.")
    parser.add_argument("-i", "--image", type=str, help="Image path")
    parser.add_argument("-o", "--output", type=str, choices=['cv2', 'plt', 'file'], default='cv2', help="Output mode: cv2, plt, or file")
    args = parser.parse_args()

    original_image: Optional[np.ndarray] = load_image(args.image)

    if original_image is None:
        return

    processed_image = None
    try:
        processed_image, num_contours = find_and_draw_contours(original_image)
        print(f"Contours: {num_contours}")

        text = f"Contours found: {num_contours}"
        cv2.putText(processed_image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        display_or_save_image(original_image, processed_image, args.output)

    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        try:
            del original_image
            if processed_image is not None:
                del processed_image
        except NameError:
            pass
        gc.collect()

if __name__ == "__main__":
    main()