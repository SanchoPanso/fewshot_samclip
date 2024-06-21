import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from fewshot_samclip.fewshot_detector import FewShotDetector
from fewshot_samclip.utils import show_anns


def main():
    detector = FewShotDetector(
        sam_checkpoint="/opt/program/weights/sam_vit_b_01ec64.pth",
        sam_model_type="vit_b",
        clip_model_name="/opt/program/weights/ViT-B-32.pt",
    )    
    # Set query images
    query_images = [
        Image.open("images/query_image_1.jpg"), 
        Image.open("images/query_image_2.jpg"),
    ]
    detector.set_queries(query_images)

    # Do detection
    image = Image.open("images/target_image.jpg")
    selected_masks = detector(image, 0.85)
    
    # Draw and save results
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(selected_masks)
    plt.axis('off')

    os.makedirs('outs', exist_ok=True)
    plt.savefig('outs/plt_show.jpg') 


if __name__ == '__main__':
    main()
