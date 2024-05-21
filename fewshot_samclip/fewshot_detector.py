from typing import List
import torch
from PIL import Image
import numpy as np
import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class FewShotDetector:
    def __init__(
            self,
            sam_model_type: str = "vit_h", 
            sam_checkpoint: str = "sam_vit_h_4b8939.pth", 
            clip_model_name: str = "ViT-B/32", 
            device: str = None):
        
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize SAM
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam = self.sam.to(device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        
        # Initialize CLIP
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        
        # Placeholder for query images
        self.query_embeddings = None

    def set_queries(self, query_images: List[Image.Image]):
        # Preprocess and get CLIP embeddings for query images
        self.query_embeddings = []
        
        for img in query_images:
            processed_image = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        
            with torch.no_grad():
                embedding = self.clip_model.encode_image(processed_image)
                self.query_embeddings.append(embedding)
        
        self.query_embeddings = torch.stack(self.query_embeddings)

    def __call__(self, image: Image.Image, threshold: float = 0.5) -> List[dict]:
        # Generate masks using SAM
        masks = self.generate_masks(np.array(image))
        
        # Filter masks based on similarities
        selected_masks = self.filter_masks(image, masks, threshold)
        
        return selected_masks

    def generate_masks(self, image: np.ndarray) -> List[dict]:
        # Generate masks using SAM
        masks = self.mask_generator.generate(image)
        return masks
    
    def filter_masks(self, image: Image.Image, masks: List[dict], threshold: float) -> List[dict]:
        # Extract masked regions and get CLIP embeddings
        mask_images = []
        for mask in masks:
            bbox = mask['bbox']
            x, y, w, h = bbox
            mask_image = image.crop((x, y, x + w, y + h))
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
        selected_masks = []
        for i, score in enumerate(aggregated_similarities):
            if score > threshold:
                selected_masks.append(masks[i])

        return selected_masks
    