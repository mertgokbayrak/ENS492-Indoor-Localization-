import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold
from tqdm import tqdm


class Frame:
    def __init__(self, data_type, room_label, sequence, file_name, depth_image_path, color_image_path, pose):
        self.data_type = data_type
        self.room_label = room_label
        self.sequence = sequence
        self.file_name = file_name
        self.depth_image_path = depth_image_path
        self.color_image_path = color_image_path
        self.pose = pose


def parse_pose_file(pose_file_path):
    with open(pose_file_path, 'r') as file:
        pose = [list(map(float, line.strip().split())) for line in file]
    return pose


def create_frame_objects(data_path, room_name, data_type):
    frames = []
    # Traverse through each sequence folder
    for seq_folder in os.listdir(data_path):
        seq_path = os.path.join(data_path, seq_folder)
        if os.path.isdir(seq_path):
            print(f"Processing sequence: {seq_folder} in {room_name} ({data_type})")
            for frame_file in os.listdir(seq_path):
                if frame_file.endswith('.color.png'):
                    frame_name = frame_file.split('.')[0]
                    depth_image_path = os.path.join(seq_path, f"{frame_name}.depth.png")
                    color_image_path = os.path.join(seq_path, f"{frame_name}.color.png")
                    pose_file_path = os.path.join(seq_path, f"{frame_name}.pose.txt")

                    if os.path.exists(depth_image_path) and os.path.exists(pose_file_path):
                        print(f"Reading frame: {frame_name}")
                        pose = parse_pose_file(pose_file_path)
                        frame = Frame(data_type, room_name, seq_folder, frame_name, depth_image_path, color_image_path,
                                      pose)
                        frames.append(frame)
    return frames


def create_data_structure(data_folder):
    train_data = []
    test_data = []
    room_names = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']

    for room_name in room_names:
        print(f"Processing room: {room_name}")
        test_path = os.path.join(data_folder, room_name, 'test')
        train_path = os.path.join(data_folder, room_name, 'train')

        train_data.extend(create_frame_objects(train_path, room_name, 'train'))
        test_data.extend(create_frame_objects(test_path, room_name, 'test'))

    return train_data, test_data


# Replace 'your_path_to_data_folder' with the path to the 'data' folder on your local system
your_path_to_data_folder = '/Volumes/MERT SSD/data'
train_data, test_data = create_data_structure(your_path_to_data_folder)

# For debug purposes: print out the first few frame paths to check
for frame in train_data[:5] + test_data[:5]:
    print(f"Room: {frame.room_label}, Sequence: {frame.sequence}, Frame: {frame.file_name}")
    print(f"Depth Image Path: {frame.depth_image_path}")
    print(f"Color Image Path: {frame.color_image_path}")
    print(f"Pose: {frame.pose}")
    print('-----------------------------------------------')


def display_image(image_path):
    with Image.open(image_path).convert('RGB') as img:
        plt.imshow(img)
        plt.axis('off')


def display_image2(image_path):
    with Image.open(image_path) as img:
        plt.imshow(img, cmap='gray')  # 'cmap' parameter specifies that we want the image in grayscale
        plt.axis('off')


def display_frame_images(frame):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"Color Image - {frame.file_name}")
    display_image(frame.color_image_path)

    plt.subplot(1, 2, 2)
    plt.title(f"Depth Image - {frame.file_name}")
    display_image2(frame.depth_image_path)

    plt.show()


for frame in train_data[:2] + test_data[:2]:
    print(
        f"Room: {frame.room_label}, Sequence: {frame.sequence}, Frame: {frame.file_name}, Data type: {frame.data_type}")
    print(f"Depth Image Path: {frame.depth_image_path}")
    print(f"Color Image Path: {frame.color_image_path}")
    print(f"Pose: {frame.pose}")
    display_frame_images(frame)
    print('-----------------------------------------------')

print(f"Total Train Frames: {len(train_data)}")
print(f"Total Test Frames: {len(test_data)}")


def check_frames(frames):
    missing_files = []
    invalid_poses = []

    for frame in frames:
        # Check for missing color or depth image files
        if not os.path.exists(frame.color_image_path) or not os.path.exists(frame.depth_image_path):
            missing_files.append(frame)
            continue  # Skip to the next frame if a file is missing

        # Check pose file for NaN values or if it doesn't exist
        try:
            pose_matrix = frame.pose
            if np.isnan(pose_matrix).any():  # Check if there are any NaN values in the pose matrix
                invalid_poses.append(frame)
        except IOError:  # This catches the case where the pose file doesn't exist
            missing_files.append(frame)
        except ValueError:  # This catches the case where the pose file has invalid contents (e.g., non-numeric values)
            invalid_poses.append(frame)

    return missing_files, invalid_poses


def report_issues(missing_files, invalid_poses, data_type):
    print(f"{data_type} Data:")
    print(f"Frames with missing color/depth images or pose.txt: {len(missing_files)}")
    for frame in missing_files:
        print(f"Missing file in frame: {frame.file_name}")

    print(f"Frames with invalid pose data: {len(invalid_poses)}")
    for frame in invalid_poses:
        print(f"Invalid pose data in frame: {frame.file_name}")
    print("---------------------------------------------------")


# Assuming train_data and test_data are your arrays of Frame objects
train_missing, train_invalid = check_frames(train_data)
test_missing, test_invalid = check_frames(test_data)

# Report the issues found
report_issues(train_missing, train_invalid, "Train")
report_issues(test_missing, test_invalid, "Test")


# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, frames, label_map, transform=None):
        self.frames = frames
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        image = Image.open(frame.color_image_path).convert('RGB')
        label = self.label_map[frame.room_label]

        if self.transform:
            image = self.transform(image)

        return image, label


transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

label_map = {label: index for index, label in
             enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs'])}
num_classes = len(label_map)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = CustomDataset(train_data, label_map, transform=transformations)
test_dataset = CustomDataset(test_data, label_map, transform=transformations)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

n_splits = 5
num_epochs = 15
dataset_size = len(train_loader.dataset)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

indices = range(dataset_size)

for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
    print("-" * 100)
    print(f"FOLD {fold}")
    print("-" * 100)

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    validation_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=64, sampler=train_subsampler)
    validation_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=64, sampler=validation_subsampler)

    model = models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    for param in model.features.parameters():
        param.requires_grad = False

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_val_loss = np.inf
    best_epoch = -1
    best_model_path = f'best_model_fold_{fold}.pth'

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs), desc="Epochs Progress"):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(validation_loader, desc=f"Validation Epoch {epoch + 1}", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(validation_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    print("-" * 50)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses for Fold {fold}')
    plt.legend()
    plt.show()

    print("-" * 50)
    print(f"Training complete. Best Epoch: {best_epoch + 1} with Validation Loss: {best_val_loss:.4f}")

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Performance of the best model from fold {fold} on the test data: Accuracy = {accuracy:.2f}%")
    print("-" * 50)
