import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from fewshot_samclip.fewshot_detector import FewShotDetector
from fewshot_samclip.utils import show_anns


def main():
    detector = FewShotDetector()
    query_images = [
        Image.open("images/query_image_1.jpg"), 
        Image.open("images/query_image_2.jpg"),
    ]
    detector.set_queries(query_images)

    image = Image.open("images/target_image.jpg")
    selected_masks = detector(image, 0.85)
    
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(selected_masks)
    plt.axis('off')

    os.makedirs('outs', exist_ok=True)
    plt.savefig('outs/plt_show.jpg') 


if __name__ == '__main__':
    main()
