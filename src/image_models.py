#%%
import os
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
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
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        
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
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        def process_image(image_path):
            features = self.extract_features(image_path)
            base_name = os.path.basename(image_path)
            class_name = os.path.basename(os.path.dirname(image_path))
            class_folder = os.path.join(save_folder, class_name)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            base_name = base_name.split('.')[0]
            feature_path = os.path.join(class_folder, base_name + '.npy')
            np.save(feature_path, features)
            print(f"Saved features for {image_path} to {feature_path}")

        for image_path in image_paths:
            process_image(image_path)
            
        #with ProcessPoolExecutor() as executor:
        #    executor.map(process_image, image_paths)
        
def fix_file(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        fixed_lines = []
        for line in lines:
            line = line.strip()  # 改行文字を削除
            if '/' not in line:
                # (クラス名).jpg を (クラス名)/(クラス名).jpg に修正
                class_name = line.split('_')[:-1]
                class_name = '_'.join(class_name)
                fixed_line = f"{class_name}/{line}"
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        with open(file_path, 'w') as file:
            for line in fixed_lines:
                file.write(line + '\n')

                
def pack_image_files(image_list, feature_folder):
    # image listの順に画像を読み込んで特徴ベクトルを取得し、torch.Tensorに変換
    features = []
    for image_path in image_list:
        feature_path = os.path.join(feature_folder, image_path.replace('.jpg', '.npy'))
        feature = np.load(feature_path)
        features.append(feature)
    features = np.array(features).squeeze()
    features = torch.tensor(features, dtype=torch.float32)
    return features

#%%
if __name__ == "__main__":
    folder_path = "../data/image/Images/"
    save_folder = "../data/image/CLIPvision/"

    image_paths = get_image_paths(folder_path)

    #extractor = ImageFeatureExtractor()
    #extractor.save_features(image_paths, save_folder)
    
    fix_file("../data/train_image_paths.txt")
    fix_file("../data/val_image_paths.txt")
    
    for split in ['train', 'val']:
        image_list_path = f"../data/{split}_image_paths.txt"
        # load txt file
        with open(image_list_path, 'r') as f:
            image_list = open(image_list_path).read().splitlines()
        features = pack_image_files(image_list, save_folder)
        torch.save(features, f"../data/{split}_X_image.pt")
    
# %%
