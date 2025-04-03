import cv2
import numpy as np
import random
import os


class AdversarialImageGenerator:
    def __init__(self):
        pass
    def erode(self, img):
        kernel = np.ones((5, 5), np.uint8) 
        img_erosion = cv2.dilate(img, kernel, iterations=1) 
        return img_erosion
    
    def morphological_blackhat(self, img):
        kernel = np.ones((5, 5), np.uint8) 
        img_erosion = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel, iterations=1) 
        inverted_image = cv2.bitwise_not(img_erosion)
        return inverted_image
    
    def ink_fade(self, img):
        ink_spill = np.zeros_like(img)
        num_blobs = 3
        for _ in range(num_blobs):
            x, y = np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0])
            radius = np.random.randint(10, 20)  
            cv2.circle(ink_spill, (x, y), radius, 255, -1)
        ink_spill = cv2.GaussianBlur(ink_spill, (3, 3), 7)
        inked_image = cv2.bitwise_or(img, ink_spill)
        return inked_image
    
    def ink_spill(self, img):
        ink_spill = np.zeros_like(img)
        num_blobs = 2
        for _ in range(num_blobs):
            x, y = np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0])
            radius = np.random.randint(10, 20)  
            cv2.circle(ink_spill, (x, y), radius, 255, -1)
        ink_spill = cv2.GaussianBlur(ink_spill, (3, 3), 7)
        ink_spill = cv2.threshold(ink_spill, 200, 255, cv2.THRESH_BINARY)[1]  # Convert to pure black and white
        ink_spill = 255 - ink_spill     
        inked_image = cv2.bitwise_and(img, ink_spill)
        return inked_image
    
    def age_parchment(self, img):
        sepia_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        text_mask = (img == 0)
        h, w = img.shape
        scaling_mask = np.ones((h, w), dtype=np.float32)

        # Define region-based scaling factors
        # Example: Divide the image into quadrants with different sepia intensities
        scaling_mask[:h//2, :w//2] = 0.97  # Top-left (lighter sepia)
        scaling_mask[:h//2, w//2:] = 1.0  # Top-right (normal sepia)
        scaling_mask[h//2:, :w//2] =  1.04  # Bottom-left (stronger sepia)
        scaling_mask[h//2:, w//2:] = 0.94  # Bottom-right (very strong sepia)
        scaling_mask = np.dstack([scaling_mask] * 3) 
        sepia_background = np.array([220, 245, 245], dtype=np.uint8)
        sepia_background = (sepia_background * scaling_mask).astype(np.uint8)
 

        sepia_image[text_mask] = (0, 0, 0)  # Keep text black
        sepia_image[~text_mask] = sepia_background[~text_mask] 
        # inverted_image = cv2.bitwise_not(sepia_final)
        return sepia_image

    def target_region(self, img):
        inverted_image = cv2.bitwise_not(img)
        contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_index = random.randint(0, len(contours) - 1)  # Change this to select a different character
        x, y, w, h = cv2.boundingRect(contours[char_index])
        char_region = img[y:y+h, x:x+w]
        return char_region, x, y, w, h

generator = AdversarialImageGenerator()
dir = '/Users/rishabmtr/GeorgiaTechCS/CS8803/CS8803-HUM/washingtondb/data/line_images_normalized/'
output_dir = '/Users/rishabmtr/GeorgiaTechCS/CS8803/CS8803-HUM/adversarial_generation/combined'
for filename in os.listdir(dir):
    img_str = os.path.join(dir, filename)
    if filename == '.DS_Store':
        continue
    img = cv2.imread(img_str, 0)
    modified_image = img.copy()
    for function in ['erode', 'morphological-blackhat', 'ink-fade', 'ink-spill', 'age-parchment']:
        for i in range(0, 3):
            char_region, x, y, w, h = generator.target_region(img)
            if function == 'erode':
                char_region = generator.erode(char_region)
            elif function == 'morphological-blackhat':
                char_region = generator.morphological_blackhat(char_region)
            elif function == 'ink-fade':
                char_region = generator.ink_fade(char_region)
            elif function == 'ink-spill':
                char_region = generator.ink_spill(char_region)
            modified_image[y:y+h, x:x+w] = char_region
    modified_image = generator.age_parchment(modified_image)
    output_filename = filename.split(".")[0]
    output_filename += '-combined.png'
    output_filename = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_filename, modified_image)








