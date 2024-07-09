#%%
import os
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
#%%

def get_image_paths(folder_path):
    """
    指定されたフォルダ内のすべての画像ファイルのパスを取得します。
    
    Args:
    - folder_path (str): 画像が格納されているフォルダのパス。
    
    Returns:
    - List[str]: 画像ファイルのパスのリスト。
    """
    image_paths = []
    for file_path in Path(folder_path).rglob('*'):
        if file_path.suffix in ['.png', '.jpg', '.jpeg']:
            image_paths.append(str(file_path))
    return image_paths


class ImageFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch16"):
        """
        CLIPモデルとプロセッサを初期化します。
        
        Args:
        - model_name (str): 使用するCLIPモデルの名前。
        """
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def extract_features(self, image_path):
        """
        画像ファイルのパスを受け取り、その画像の内部表現を取得します。
        
        Args:
        - image_path (str): 画像ファイルのパス。
        
        Returns:
        - np.ndarray: 画像の特徴ベクトル。
        """
        img = Image.open(image_path)
        inputs = self.processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        
        return features.cpu().numpy()

    def save_features(self, image_paths, save_folder):
        """
        画像ファイルのパスを受け取り、その画像の内部表現を取得し、指定されたフォルダに保存します。
        
        Args:
        - image_paths (List[str]): 画像ファイルのパスのリスト。
        - save_folder (str): 特徴ベクトルを保存するフォルダのパス。
        """
        for image_path in image_paths:
            features = self.extract_features(image_path)
            base_name = os.path.basename(image_path)
            class_name = os.path.basename(os.path.dirname(image_path))
            class_folder = os.path.join(save_folder, class_name)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            feature_path = os.path.join(class_folder, base_name + '.npy')
            np.save(feature_path, features)
            print(f"Saved features for {image_path} to {feature_path}")


#%%
if __name__ == "__main__":
    folder_path = "../data/image/Images/"
    save_folder = "../data/image/CLIPvision/"

    image_paths = get_image_paths(folder_path)

    extractor = ImageFeatureExtractor()
    extractor.save_features(image_paths, save_folder)

# %%
