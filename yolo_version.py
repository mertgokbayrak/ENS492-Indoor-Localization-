import device
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import torch.optim as optim


class Frame:
    def __init__(self, data_type, room_label, sequence, file_name, depth_image_path, color_image_path, pose, boxes,
                 labels):
        self.data_type = data_type
        self.room_label = room_label
        self.sequence = sequence
        self.file_name = file_name
        self.depth_image_path = depth_image_path
        self.color_image_path = color_image_path
        self.pose = pose
        self.boxes = boxes  # This should be a list of [x_center, y_center, width, height]
        self.labels = labels  # This should be a list of integers representing class labels


def parse_pose_file(pose_file_path):
    with open(pose_file_path, 'r') as file:
        pose = [list(map(float, line.strip().split())) for line in file]
    return pose


def parse_annotation_file(annotation_path):
    boxes = []
    labels = []
    with open(annotation_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            label = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            boxes.append([x_center, y_center, width, height])
            labels.append(label)
    return boxes, labels


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
                    # annotation_path = os.path.join(seq_path,f"{frame_name}.txt")  # Assuming annotations are in a
                    # .txt file

                    if os.path.exists(color_image_path) and os.path.exists(pose_file_path):
                        pose = parse_pose_file(pose_file_path)
                        # boxes, labels = parse_annotation_file(annotation_path)
                        frame = Frame(data_type, room_name, seq_folder, frame_name, None, color_image_path, pose)
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


def display_image_in_grayscale(image_path):
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
    display_image_in_grayscale(frame.depth_image_path)

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


class CustomDataset(Dataset):
    def __init__(self, frames, transform=None):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        image = Image.open(frame.color_image_path).convert('RGB')

        # Prepare data for YOLO
        targets = {'boxes': torch.Tensor(frame.boxes), 'labels': torch.LongTensor(frame.labels)}

        # Pose data
        pose = torch.tensor(frame.pose, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, targets, pose


def calculate_iou(pred_boxes, gt_boxes):
    """
    Calculate the IoU between predicted and ground truth boxes where boxes are in (x_center, y_center, width, height).
    """
    # Convert from center coordinates to bounding box corners
    pred_boxes = torch.stack([
        pred_boxes[:, 0] - pred_boxes[:, 2] / 2,  # xmin
        pred_boxes[:, 1] - pred_boxes[:, 3] / 2,  # ymin
        pred_boxes[:, 0] + pred_boxes[:, 2] / 2,  # xmax
        pred_boxes[:, 1] + pred_boxes[:, 3] / 2   # ymax
    ], dim=1)

    gt_boxes = torch.stack([
        gt_boxes[:, 0] - gt_boxes[:, 2] / 2,
        gt_boxes[:, 1] - gt_boxes[:, 3] / 2,
        gt_boxes[:, 0] + gt_boxes[:, 2] / 2,
        gt_boxes[:, 1] + gt_boxes[:, 3] / 2
    ], dim=1)

    # Intersection
    inter_rect_x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    inter_rect_y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    inter_rect_x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    inter_rect_y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # Union Area
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    union_area = pred_area + gt_area - inter_area

    # IoU
    iou = inter_area / union_area

    return iou


def evaluate_model(test_loader, model, pose_model, device):
    model.eval()
    pose_model.eval()
    total_detection_loss = 0
    total_pose_loss = 0
    total_correct_detections = 0
    total_detections = 0

    with torch.no_grad():
        for images, targets, poses in test_loader:
            images = images.to(device)
            poses = poses.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Object detection with YOLO
            results = model(images)
            detections = results.xyxy[0]  # Assuming xyxy format
            detection_loss = sum([t['loss'] for t in results])  # Aggregate loss if provided directly

            # Pose estimation
            pose_outputs = pose_model(images)
            pose_loss = pose_criterion(pose_outputs, poses)

            total_detection_loss += detection_loss.item()
            total_pose_loss += pose_loss.item()

            # Simplified accuracy based on IoU > 0.5
            for i, det in enumerate(detections):
                ground_truth_boxes = targets[i]['boxes']
                iou_scores = calculate_iou(det[:, :4], ground_truth_boxes)
                total_correct_detections += (iou_scores > 0.5).sum().item()  # Counting detections with IoU > 0.5
                total_detections += ground_truth_boxes.size(0)

    average_detection_loss = total_detection_loss / len(test_loader)
    average_pose_loss = total_pose_loss / len(test_loader)
    detection_accuracy = 100 * total_correct_detections / total_detections if total_detections > 0 else 0

    print(f"Average Detection Loss: {average_detection_loss:.4f}")
    print(f"Average Pose Loss: {average_pose_loss:.4f}")
    print(f"Detection Accuracy: {detection_accuracy:.2%}")
    print("-" * 50)


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

pose_model = models.resnet18(pretrained=True)
pose_model.fc = nn.Linear(pose_model.fc.in_features, 6)  # 6 values (x, y, z, yaw, pitch, roll)
pose_model.to(device)

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640)),  # Resize images to the input size expected by YOLOv5
])

train_dataset = CustomDataset(train_data, transform=transformations)
test_dataset = CustomDataset(test_data, transform=transformations)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        batch_size=32, sampler=train_subsampler)
    validation_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=32, sampler=validation_subsampler)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_yolo = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_val_loss = np.inf
    best_epoch = -1
    best_model_path = f'yolo_best_model_fold_{fold}.pth'

    train_losses = []
    val_losses = []

    # posenet initialization start
    pose_model = models.resnet18(pretrained=True)
    pose_model.fc = nn.Linear(pose_model.fc.in_features, 6)
    pose_model.to(device)

    pose_criterion = nn.MSELoss()
    optimizer_pose = optim.SGD(list(model.parameters()) + list(pose_model.parameters()), lr=0.001, momentum=0.9)
    # posenet initialization end

    for epoch in tqdm(range(num_epochs), desc="Epochs Progress"):
        model.train()  # Set YOLO model to training mode
        pose_model.train()  # Set Pose model to training mode
        train_loss = 0.0  # Initialize the loss accumulator for object detection
        pose_loss = 0.0  # Initialize the loss accumulator for pose estimation

        for images, targets, poses in train_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            poses = poses.to(device)

            # Clear gradients for both optimizers
            optimizer_yolo.zero_grad()
            optimizer_pose.zero_grad()

            # Compute loss for YOLO
            results = model(images, targets)
            loss_detection = results['loss']
            loss_detection.backward(retain_graph=True)

            # Compute loss for Pose Model
            pose_outputs = pose_model(images)
            loss_pose = pose_criterion(pose_outputs, poses)
            loss_pose.backward()

            # Update parameters
            optimizer_yolo.step()  # Update YOLO model parameters
            optimizer_pose.step()  # Update Pose Model parameters

            # Accumulate losses for reporting
            train_loss += loss_detection.item() * images.size(0)  # Scale loss by the batch size
            pose_loss += loss_pose.item() * images.size(0)  # Scale loss by the batch size

        # Calculate average losses for the epoch
        train_loss /= len(train_loader.dataset)
        pose_loss /= len(train_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs} - Detection Loss: {train_loss:.4f}, Pose Loss: {pose_loss:.4f}")
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

    evaluate_model(test_loader, model, pose_model, device)



