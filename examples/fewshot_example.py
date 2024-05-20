from PIL import Image
import matplotlib.pyplot as plt
from fewshot_samclip.fewshot_detector import FewShotDetector
from fewshot_samclip.utils import show_anns


def main():
    detector = FewShotDetector()
    query_images = [
        Image.open("images/query_image.jpg"), 
        # Image.open("path_to_query_image2.jpg")
    ]
    detector.set_queries(query_images)

    image = Image.open("images/target_image.jpg")
    selected_masks = detector(image, 0.95)
    
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(selected_masks)
    plt.axis('off')
    plt.savefig('plt_show.jpg') 


if __name__ == '__main__':
    main()
