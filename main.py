import os
from PIL import Image
import numpy as np

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
                        frame = Frame(data_type, room_name, seq_folder, frame_name, depth_image_path, color_image_path, pose)
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
your_path_to_data_folder = 'data'
train_data, test_data = create_data_structure(your_path_to_data_folder)

# For debug purposes: print out the first few frame paths to check
for frame in train_data[:5] + test_data[:5]:
    print(f"Room: {frame.room_label}, Sequence: {frame.sequence}, Frame: {frame.file_name}")
    print(f"Depth Image Path: {frame.depth_image_path}")
    print(f"Color Image Path: {frame.color_image_path}")
    print(f"Pose: {frame.pose}")
    print('-----------------------------------------------')

print(f"Total Train Frames: {len(train_data)}")
print(f"Total Test Frames: {len(test_data)}")


