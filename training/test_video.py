import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import encoder
import cv2
import numpy as np
import time
import random
def test():
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Step")
    # Initialize model
    autoencoder = encoder.AutoEncoder(in_channels=3, out_channels=1).to(device)
    print("Step")
    # Specify the path to the saved model
    model_path = "./saved_model"
    print("Step")
    # Load the state dict previously saved
    state_dict = torch.load(model_path, map_location=device)
    print("Step")
    # Load the state dict to the model
    autoencoder.load_state_dict(state_dict)

    # Make sure the model is in evaluation mode
    autoencoder.eval()
    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for pretrained PyTorch models
    ])

    tf=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512,640)),
    ])
    # Specify the path to your video file
    video_path = "/home/tanujthakkar/ash/PRG-Hoop-Recognition/assets/IMG_8624.MOV"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('../video/output1.mp4', fourcc, 30.0, (1920, 512))
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    THRESHOLD = 0.5
    if not cap.isOpened():
        print("Error")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                break

            frame = tf(frame)
            frame = frame.permute(1,2,0)
            #cv2.imshow('Frame', frame.numpy())
            
            pred = frame.permute(2,0,1)
            pred = pred.unsqueeze(0)
            pred = pred.to('cuda')
            pred = autoencoder(pred)

            pred = pred.squeeze()
            pred = pred.cpu()
            pred = pred.detach()
            pred = (pred > THRESHOLD).type(torch.float)
            #cv2.imshow('Mask', pred.numpy())

            frame_np = frame.cpu().detach().numpy()
            pred_np = pred.cpu().detach().numpy()
            mask = (pred_np > THRESHOLD).astype(np.uint8)
            color_mask = np.zeros_like(frame_np)

            color_mask[mask == 1] = [0, 255, 0]

            overlayed_frame = cv2.addWeighted(frame_np, 1, color_mask, 0.5, 0)


            #display all 3
            frame_np = frame.cpu().detach().numpy()

# Assuming 'pred' is a binary mask with values 0 or 1 of shape [512, 640]
            mask_np = pred.cpu().detach().numpy()
            mask_binary = (mask_np > THRESHOLD).astype(np.uint8)  # Thresholding to create a binary mask
            mask_3_channel = np.stack((mask_binary, mask_binary, mask_binary), axis=-1) * 255

            # Combine frame, mask, and overlayed_frame horizontally
            combined_frame = np.hstack((frame_np, mask_3_channel, overlayed_frame))
            cv2.imshow('Result', combined_frame)
            time.sleep(0.005)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    test()