import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class Frame:
    def __init__(self, room_label, sequence, file_name, color_image_path, pose):
        self.room_label = room_label
        self.sequence = sequence
        self.file_name = file_name
        self.color_image_path = color_image_path
        self.pose = pose


def parse_pose_file(pose_file_path):
    with open(pose_file_path, 'r') as file:
        pose = np.array([list(map(float, line.strip().split())) for line in file]).flatten()
    return pose


def create_frame_objects(data_path, room_name, data_type):
    frames = []
    for seq_folder in os.listdir(data_path):
        seq_path = os.path.join(data_path, seq_folder)
        if os.path.isdir(seq_path):
            for frame_file in os.listdir(seq_path):
                if frame_file.endswith('.color.png'):
                    frame_name = frame_file.split('.')[0]
                    color_image_path = os.path.join(seq_path, f"{frame_name}.color.png")
                    pose_file_path = os.path.join(seq_path, f"{frame_name}.pose.txt")
                    if os.path.exists(color_image_path) and os.path.exists(pose_file_path):
                        pose = parse_pose_file(pose_file_path)
                        frame = Frame(room_name, seq_folder, frame_name, color_image_path, pose)
                        frames.append(frame)
    return frames


def create_data_structure(data_folder):
    train_data = []
    test_data = []
    room_names = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
    for room_name in room_names:
        train_path = os.path.join(data_folder, room_name, 'train')
        test_path = os.path.join(data_folder, room_name, 'test')
        train_data.extend(create_frame_objects(train_path, room_name, 'train'))
        test_data.extend(create_frame_objects(test_path, room_name, 'test'))
    return train_data, test_data


your_path_to_data_folder = '/Volumes/MERT SSD/data'
train_data, test_data = create_data_structure(your_path_to_data_folder)


class CustomDataset(Dataset):
    def __init__(self, frames, transform=None):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        image = Image.open(frame.color_image_path).convert('RGB')
        pose = np.array(frame.pose, dtype=np.float32).reshape(4, 4).flatten()  # Reshape pose data into a 4x4 matrix

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(pose)  # Return pose as a 4x4 tensor


transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CustomDataset(train_data, transform=transformations)
test_dataset = CustomDataset(test_data, transform=transformations)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pose_model = models.resnet18(pretrained=True)
pose_model.fc = nn.Linear(pose_model.fc.in_features, 16)  # Adjusting the final layer to predict 6 values for the pose
pose_model.to(device)

optimizer = optim.SGD(pose_model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.MSELoss()

for epoch in tqdm(range(15), desc="Epochs Progress"):  # Assuming 15 epochs
    pose_model.train()
    train_loss = 0.0
    for images, poses in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False):
        images, poses = images.to(device), poses.to(device)
        optimizer.zero_grad()
        outputs = pose_model(images)
        loss = criterion(outputs, poses)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}, Average Training Loss: {train_loss:.4f}")

    pose_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, poses in tqdm(test_loader, desc=f"Evaluating Epoch {epoch + 1}", leave=False):
            images, poses = images.to(device), poses.to(device)
            outputs = pose_model(images)
            loss = criterion(outputs, poses)
            total_loss += loss.item() * images.size(0)

    average_loss = total_loss / len(test_loader.dataset)
    print(f"Epoch {epoch+1}, Average Validation Loss: {average_loss:.4f}")
    # Epoch 15, Average Training Loss: 0.0110 = %1.1
    # Epoch 15, Average Validation Loss: 0.0401 = %4.01
