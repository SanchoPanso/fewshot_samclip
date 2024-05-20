import torch
from PIL import Image
import numpy as np
import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class FewShotDetector:
    def __init__(
            self,
            sam_model_type: str, 
            sam_checkpoint: str, 
            clip_model_name: str = "ViT-B/32", 
            device: str = None):
        
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize SAM
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        
        # Initialize CLIP
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        
        # Placeholder for query images
        self.query_embeddings = None

    def set_queries(self, query_images: list):
        # Preprocess and get CLIP embeddings for query images
        self.query_embeddings = []
        for img in query_images:
            processed_image = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.encode_image(processed_image)
                self.query_embeddings.append(embedding)
        self.query_embeddings = torch.stack(self.query_embeddings)

    def __call__(self, image: Image):
        # Generate masks using SAM
        masks = self.mask_generator.generate(image)
        
        # Extract masked regions and get CLIP embeddings
        mask_images = []
        for mask in masks:
            bbox = mask['bbox']
            mask_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            mask_images.append(mask_image)

        mask_embeddings = []
        for mask_image in mask_images:
            processed_image = self.clip_preprocess(mask_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.encode_image(processed_image)
                mask_embeddings.append(embedding)
        mask_embeddings = torch.cat(mask_embeddings)

        # Compare mask embeddings with query embeddings using cosine similarity
        similarities = []
        for query_embedding in self.query_embeddings:
            similarity = torch.nn.functional.cosine_similarity(query_embedding, mask_embeddings)
            similarities.append(similarity)
        
        # Aggregate similarities (e.g., mean similarity across all queries)
        aggregated_similarities = torch.stack(similarities).mean(dim=0)
        
        # Filter masks based on similarity scores
        threshold = 0.5  # You can tune this threshold
        selected_masks = [masks[i] for i, score in enumerate(aggregated_similarities) if score > threshold]
        
        return selected_masks

# Example usage:
# detector = FewShotDetector(sam_model_type="sam_vit_h", sam_checkpoint="sam_vit_h_4b8939.pth")
# query_images = [Image.open("path_to_query_image1.jpg"), Image.open("path_to_query_image2.jpg")]
# detector.set_queries(query_images)
# image = Image.open("path_to_target_image.jpg")
# selected_masks = detector(image)
# for mask in selected_masks:
#     print(mask)
