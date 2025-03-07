import torch
import clip

class CLIPModel:
    def __init__(self, class_names, device="cuda"):
        """
        Initializes the CLIP model and prepares text embeddings for class labels.

        Args:
            class_names (list): List of class names.
            device (str): Device to run the model ("cuda" or "cpu").
        """
        self.device = device if torch.cuda.is_available() else "cpu"

        # Load CLIP model
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.class_names = class_names  # Store class names

        # Tokenize class labels
        self.text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(self.device)

        # Precompute class embeddings (for efficiency)
        with torch.no_grad():
            self.text_embeddings = self.model.encode_text(self.text_inputs)
            self.text_embeddings /= self.text_embeddings.norm(dim=-1, keepdim=True)  # Normalize

    def get_clip_embeddings(self):
        """
        Returns precomputed CLIP text embeddings.

        Returns:
            torch.Tensor: Normalized CLIP text embeddings (num_classes, d_model).
        """
        return self.text_embeddings.to(torch.float32)
