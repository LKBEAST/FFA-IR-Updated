import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

resnet101 = models.resnet101(pretrained=True)
resnet101 = torch.nn.Sequential(*(list(resnet101.children())[:-1]))
resnet101.eval()

def extract_features(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")  # Convert to RGB if it's not
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = resnet101(image)
    return features.squeeze().numpy()


main_directory = 'FFAIR' 
feature_dict = {}

case_names = [case for case in os.listdir(main_directory) if case.startswith('case_')]

print(f"Processing {len(case_names)} cases...")

for case_name in tqdm(case_names): # Progress bar for cases
    print(f"Processing {case_name}...")
    case_directory = os.path.join(main_directory, case_name)
    case_features = []
    image_names = [img for img in os.listdir(case_directory) if img.endswith('.jpeg')]
    for image_name in tqdm(image_names, desc=f'Images in {case_name}'): # Progress bar for images
        image_path = os.path.join(case_directory, image_name)
        features = extract_features(image_path)
        case_features.append(features)
    feature_dict[case_name] = np.stack(case_features)
    print(f"Finished processing {case_name}")

# Saving each case's features as separate .npy files
print("Saving features...")
for case_name, case_features in feature_dict.items():
    np.save(f'{case_name}_features.npy', case_features)
    print(f"Saved {case_name}_features.npy")

print("Processing complete!")
