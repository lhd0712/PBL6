# extract_features.py (Phiên bản 10-Crop CHUẨN + Resumable An toàn)
import torch
import cv2
import numpy as np
import os
import glob
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    FiveCropVideo
)
from pytorchvideo.transforms.functional import hflip
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    NormalizeVideo,
)

# --- 1. Thiết lập Mô hình I3D (Giữ nguyên) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_i3d = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
model_i3d = model_i3d.to(device)
model_i3d.eval()

class I3DFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.blocks = model.blocks
    def forward(self, x):
        for i in range(len(self.blocks) - 1): # Chạy hết trừ lớp head cuối
             x = self.blocks[i](x)
        return x.squeeze() # Lấy feature từ avg_pool

feature_extractor = I3DFeatureExtractor(model_i3d)

# --- 2. Thiết lập Pre-processing (Giữ nguyên) ---
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
num_frames_to_sample = 16 
clip_duration = 1.0 
crop_size = 224 

base_transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames_to_sample),
            Lambda(lambda x: x / 255.0), 
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
        ]
    ),
)
five_crop_transform = FiveCropVideo(size=(crop_size, crop_size))

# --- 3. Vòng lặp Trích xuất (Logic Resumable được nâng cấp) ---
# !!! THAY ĐỔI ĐƯỜNG DẪN CỦA BẠN Ở ĐÂY !!!
VIDEO_ROOT = "UCF_Crimes_Videos"
FEATURE_ROOT = "I3D_Features_10crop"

video_paths = glob.glob(os.path.join(VIDEO_ROOT, "*", "*.mp4"))
print(f"Tìm thấy {len(video_paths)} video để xử lý (10-crop mode).")

processed_count = 0
skipped_count = 0

for video_path in video_paths:
    
    # Tạo đường dẫn lưu (save_path) TRƯỚC TIÊN
    save_dir = video_path.replace(VIDEO_ROOT, FEATURE_ROOT).rsplit(os.sep, 1)[0]
    save_path = video_path.replace(VIDEO_ROOT, FEATURE_ROOT).replace(".mp4", ".npy")

    # =========================================================
    # === TÍNH NĂNG RESUMABLE (TIẾP TỤC) ===
    # Kiểm tra xem tệp đã tồn tại VÀ có kích thước > 1KB (tránh tệp hỏng)
    file_exists = os.path.exists(save_path)
    file_is_valid = file_exists and os.path.getsize(save_path) > 1024 # > 1KB

    if file_is_valid:
        print(f"Đã tồn tại (hợp lệ): {video_path}, bỏ qua.")
        skipped_count += 1
        continue  # Đi đến video tiếp theo
    elif file_exists:
        # Tệp tồn tại nhưng 0KB (bị hỏng), xóa đi để làm lại
        print(f"Tồn tại tệp hỏng (0KB): {save_path}. Xóa và làm lại.")
        os.remove(save_path)
    # =========================================================
    
    # Nếu tệp không tồn tại, hoặc bị hỏng, chúng ta mới bắt đầu xử lý
    print(f"Đang xử lý (10-crop): {video_path}")
    os.makedirs(save_dir, exist_ok=True)

    try:
        video_features_10crop_avg = []
        video = EncodedVideo.from_path(video_path)
        video_duration = video.duration
        
        for start_sec in range(0, int(video_duration), 1):
            clip_data = video.get_clip(start_sec=start_sec, end_sec=start_sec + clip_duration)
            clip_data_base = base_transform(clip_data)
            base_tensor = clip_data_base["video"]
            
            # Logic 10-CROP
            five_clips_orig = five_crop_transform(base_tensor)
            base_tensor_flipped = hflip(base_tensor)
            five_clips_flipped = five_crop_transform(base_tensor_flipped)
            all_10_clips = torch.stack(five_clips_orig + five_clips_flipped, dim=0).to(device)

            with torch.no_grad():
                features_10_crops = feature_extractor(all_10_clips)
                features_avg = torch.mean(features_10_crops, dim=0)
                video_features_10crop_avg.append(features_avg.cpu().numpy())

        if not video_features_10crop_avg:
            print(f"Cảnh báo: Video {video_path} quá ngắn.")
            continue
            
        video_features_np = np.array(video_features_10crop_avg)
        
        # Lưu ra tệp .npy
        np.save(save_path, video_features_np)
        processed_count += 1
        
    except Exception as e:
        print(f"LỖI khi xử lý {video_path}: {e}.")
        # Nếu lỗi, xóa tệp .npy hỏng (nếu có) để lần sau chạy lại
        if os.path.exists(save_path):
            os.remove(save_path)

print("--- HOÀN TẤT GIAI ĐOẠN 1 (10-CROP) ---")
print(f"Đã xử lý mới: {processed_count} video.")
print(f"Đã bỏ qua (đã có): {skipped_count} video.")