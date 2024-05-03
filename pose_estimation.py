import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import KFold


class Frame:
    def __init__(self, data_type, room_label, sequence, file_name, color_image_path, pose):
        self.data_type = data_type
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
            print(f"Processing sequence: {seq_folder} in {room_name} ({data_type})")
            for frame_file in os.listdir(seq_path):
                if frame_file.endswith('.color.png'):
                    frame_name = frame_file.split('.')[0]
                    color_image_path = os.path.join(seq_path, f"{frame_name}.color.png")
                    pose_file_path = os.path.join(seq_path, f"{frame_name}.pose.txt")
                    if os.path.exists(color_image_path) and os.path.exists(pose_file_path):
                        pose = parse_pose_file(pose_file_path)
                        frame = Frame(data_type, room_name, seq_folder, frame_name, color_image_path, pose)
                        frames.append(frame)
    return frames


def create_data_structure(data_folder):
    local_train_data = []
    local_test_data = []
    room_names = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
    for room_name in room_names:
        train_path = os.path.join(data_folder, room_name, 'train')
        test_path = os.path.join(data_folder, room_name, 'test')
        local_train_data.extend(create_frame_objects(train_path, room_name, 'train'))
        local_test_data.extend(create_frame_objects(test_path, room_name, 'test'))
    return local_train_data, local_test_data


def create_data_structure_for_each_scene(data_folder, room_name):
    train_path = os.path.join(data_folder, room_name, 'train')
    test_path = os.path.join(data_folder, room_name, 'test')
    local_train_data = create_frame_objects(train_path, room_name, 'train')
    local_test_data = create_frame_objects(test_path, room_name, 'test')
    return local_train_data, local_test_data


your_path_to_data_folder = '/Volumes/MERT SSD/data'
# train_data, test_data = create_data_structure(your_path_to_data_folder)


class CustomDataset(Dataset):
    def __init__(self, frames, transform=None):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        image = Image.open(frame.color_image_path).convert('RGB')
        pose_matrix = np.array(frame.pose, dtype=np.float32).reshape(4, 4)
        translation = pose_matrix[:3, 3]
        rotation = pose_matrix[:3, :3]

        if self.transform:
            image = self.transform(image)
        return image, torch.from_numpy(translation), torch.from_numpy(rotation.flatten())


transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# train_dataset = CustomDataset(train_data, transform=transformations)
# test_dataset = CustomDataset(test_data, transform=transformations)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PoseModel(nn.Module):
    def __init__(self):
        super(PoseModel, self).__init__()
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)
        self.fc_translation = nn.Linear(self.backbone.fc.in_features, 3)
        self.fc_rotation = nn.Linear(self.backbone.fc.in_features, 9)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        features = self.backbone(x)
        translation = self.fc_translation(features)
        rotation = self.fc_rotation(features)
        return translation, rotation


pose_model = PoseModel().to(device)
optimizer = optim.SGD(pose_model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.MSELoss()


def rotation_matrix_to_angle_axis(rotation_matrices):
    """Convert a batch of rotation matrices to angle-axis vectors."""
    # Calculate the trace of each 3x3 rotation matrix in the batch
    traces = torch.einsum('bii->b', rotation_matrices)  # Sum over the diagonal elements in each matrix in the batch
    cos_thetas = (traces - 1) / 2.0
    cos_thetas = torch.clamp(cos_thetas, -1, 1)  # Numerical errors might make cos(theta) slightly out of its range
    thetas = torch.acos(cos_thetas)  # Angles

    # Initialize angle-axis vectors
    angle_axes = torch.zeros_like(rotation_matrices[:, :, 0])

    # Compute sin(theta) for normalization
    sin_thetas = torch.sin(thetas)

    # Find indices where theta is not too small (to avoid division by zero)
    valid = sin_thetas > 1e-5

    # For valid indices where theta is not too small, calculate angle-axis vectors
    angle_axes[valid] = torch.stack([
        rotation_matrices[valid, 2, 1] - rotation_matrices[valid, 1, 2],
        rotation_matrices[valid, 0, 2] - rotation_matrices[valid, 2, 0],
        rotation_matrices[valid, 1, 0] - rotation_matrices[valid, 0, 1]
    ], dim=1) / (2 * sin_thetas[valid].unsqueeze(1)) * thetas[valid].unsqueeze(1)

    return angle_axes


def rotation_error(pred_rot, gt_rot):
    """Calculate the angular distance between two rotation matrices."""
    pred_rot_matrix = pred_rot.view(-1, 3, 3)
    gt_rot_matrix = gt_rot.view(-1, 3, 3)
    r_diff = torch.matmul(pred_rot_matrix, gt_rot_matrix.transpose(1, 2))  # Relative rotation
    angle_axis = rotation_matrix_to_angle_axis(r_diff)
    return torch.norm(angle_axis, dim=1)  # Returns the magnitude of the angle-axis vector


def calculate_translation_error(pred, target):
    return torch.norm(pred - target, dim=1).mean()


# dataset_size = len(train_loader.dataset)
# best_fold = 0
# best_loss = 0.0
# best_model_state = None  # To store the state of the best model
# n_splits = 5
# num_epochs = 12
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
# indices = range(len(train_loader.dataset)) # warning is disregarded, since the code works correct

scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']

for scene in scenes:
    train_data, test_data = create_data_structure_for_each_scene(your_path_to_data_folder, scene)

    train_dataset = CustomDataset(train_data, transform=transformations)
    test_dataset = CustomDataset(test_data, transform=transformations)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    best_fold = 0
    best_model_state = None

    n_splits = 5
    num_epochs = 12
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = range(len(train_loader.dataset))  # warning is disregarded, since the code works correct

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print("-" * 100)
        print(f"FOLD {fold} for {scene}")
        print("-" * 100)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        validation_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        current_train_loader = torch.utils.data.DataLoader(
            train_loader.dataset, batch_size=64, sampler=train_subsampler)
        current_validation_loader = torch.utils.data.DataLoader(
            train_loader.dataset, batch_size=64, sampler=validation_subsampler)

        best_loss = np.inf
        best_epoch = -1
        for epoch in tqdm(range(num_epochs), desc=f"Epochs Progress for {scene}"):
            pose_model.train()
            total_loss = 0.0
            total_translation_error = 0.0
            total_rotation_error = 0.0

            # Training loop
            for images, translations, rotations in tqdm(current_train_loader,
                                                        desc=f"Training Epoch {epoch + 1} for {scene}", leave=False):
                images = images.to(device)
                translations = translations.to(device)
                rotations = rotations.view(-1, 3, 3).to(device)

                optimizer.zero_grad()
                trans_pred, rot_pred = pose_model(images)
                rot_pred = rot_pred.view(-1, 3, 3)

                loss_translation = criterion(trans_pred, translations)
                loss_rotation = criterion(rot_pred.view(-1, 9), rotations.view(-1, 9))
                loss = loss_translation + loss_rotation
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                translation_error = calculate_translation_error(trans_pred, translations)
                total_translation_error += translation_error.item()
                rotation_error_batch = rotation_error(rot_pred, rotations).mean().item()
                total_rotation_error += rotation_error_batch

            # Validation loop
            pose_model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for images, translations, rotations in tqdm(current_validation_loader,
                                                            desc=f"Validating Epoch {epoch + 1}",
                                                            leave=False):
                    images = images.to(device)
                    translations = translations.to(device)
                    rotations = rotations.view(-1, 3, 3).to(device)

                    trans_pred, rot_pred = pose_model(images)
                    rot_pred = rot_pred.view(-1, 3, 3)

                    loss_translation = criterion(trans_pred, translations)
                    loss_rotation = criterion(rot_pred.view(-1, 9), rotations.view(-1, 9))
                    loss = loss_translation + loss_rotation
                    validation_loss += loss.item()

            validation_loss /= len(current_validation_loader)

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_model_state = pose_model.state_dict()  # Save model state, not the model itself
                best_fold = fold
                print()
                print(f"New best model found for {scene} in fold {fold} with validation loss {best_loss:.4f}")

            average_loss = total_loss / len(train_loader)
            average_rotation_error = total_rotation_error / len(train_loader)
            average_translation_error = total_translation_error / len(train_loader)
            print(f' For {scene} in Epoch {epoch + 1}, Average Loss: {average_loss:.4f}, Average Translation Error: '
                  f'{average_translation_error:.4f}, Average Rotation Error (radians): {average_rotation_error:.4f}')

        print("-" * 50)
        print(f"Training complete for {scene} in Fold {fold}.")
        print(f"(Best Epoch for Fold {fold}: {best_epoch + 1} with Validation Loss: {best_loss:.4f}")

    # After all folds, save the best model state
    if best_model_state:
        torch.save(best_model_state, f'best_pose_model_{scene}.pth')
        print(f"Best model for {scene} from Fold {best_fold} saved with loss {best_loss:.4f}")

    # To use the best model
    pose_model.load_state_dict(torch.load(f'best_pose_model_{scene}.pth'))
    pose_model.eval()

    # Initialize metrics
    total_translation_error = 0.0
    total_rotation_error = 0.0
    count = 0

    # No gradient needed for evaluation
    with torch.no_grad():
        for images, translations, rotations in test_loader:
            images = images.to(device)
            translations = translations.to(device)
            rotations = rotations.view(-1, 3, 3).to(device)  # 3x3 rotation matrices

            # Predict
            trans_pred, rot_pred = pose_model(images)
            rot_pred = rot_pred.view(-1, 3, 3)

            # Calculate errors
            translation_error = calculate_translation_error(trans_pred, translations)
            rotation_error_batch = rotation_error(rot_pred, rotations).mean().item()

            # Aggregate errors
            total_translation_error += translation_error.item()
            total_rotation_error += rotation_error_batch
            count += 1

        # Calculate average errors
        average_translation_error = total_translation_error / count
        average_rotation_error = total_rotation_error / count

        print(f"Performance of the best model for {scene} on the test data was from Fold {best_fold}:")
        print(f"Average Translation Error for {scene}: {average_translation_error:.4f} meters")
        print(f"Average Rotation Error (radians) for {scene}: {average_rotation_error:.4f}")
        print("-" * 50)

# Results for training the model on the whole dataset
# Epoch 15, Average Training Loss: 0.0110 = 1.1%
# Epoch 15, Average Validation Loss: 0.0401 = 4.01%
# Epoch 10:
# Average Loss: 0.0392 = 3.9%,
# Average Translation Error: 0.2662 meters,
# Average Rotation Error (radians): 0.0880 = 5.04 degrees
