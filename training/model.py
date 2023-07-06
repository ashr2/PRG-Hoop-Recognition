import torch
import torchvision.transforms as transforms
import encoder
import cv2
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize model
autoencoder = encoder.AutoEncoder(in_channels=3, out_channels=1).to(device)

# Specify the path to the saved model
model_path = "./saved_model"

# Load the state dict previously saved
state_dict = torch.load(model_path, map_location=device)

# Load the state dict to the model
autoencoder.load_state_dict(state_dict)

# Make sure the model is in evaluation mode
autoencoder.eval()

# Define the transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for pretrained PyTorch models
])

# Specify the path to your video file
video_path = "/home/tanujthakkar/ash/PRG-Hoop-Recognition/assets/IMG_8621.MOV"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Iterate over the video frames
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if ret:
        # Convert image to tensor and normalize it
        image_tensor = transform(frame).unsqueeze(0).to(device)

        pred_mask = autoencoder(image_tensor)

        # Convert the tensors back to numpy arrays
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred_mask = pred_mask.squeeze().detach().cpu().numpy()

        # Optional step: Threshold the mask
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        # Convert grayscale mask to 3-channel image
        pred_mask_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

        # Visualize
        vis_stack = np.hstack((frame, pred_mask_rgb))
        vis_stack = cv2.resize(vis_stack, (int(vis_stack.shape[1] / 4), int(vis_stack.shape[0] / 4)))
        cv2.imshow('Frame', vis_stack)

        # If you press Q on your keyboard, the video will stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object
cap.release()
