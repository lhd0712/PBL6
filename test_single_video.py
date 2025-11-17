import torch
import torch.nn as nn
import numpy as np
import cv2 # Dùng OpenCV
from pathlib import Path
import time
import sys # Để lấy tham số dòng lệnh
from tqdm import tqdm # Để xem tiến trình

# Import các thư viện torchvision cần thiết
import torchvision.transforms as T
import torchvision.transforms.functional as F

# Import các model RTFM của bạn
try:
    from rtfm_model import RTFM_TFM, Classifier
except ImportError:
    print("LỖI: Không tìm thấy file rtfm_model.py. Hãy để nó chung thư mục.")
    exit()

# =========================================================
# === (QUAN TRỌNG) CẤU HÌNH ===
# =========================================================
TFM_MODEL_PATH = "rtfm_tfm_model.pth"
CLASSIFIER_MODEL_PATH = "rtfm_classifier_model.pth"
# =========================================================

# Biến toàn cục
device = "cuda" if torch.cuda.is_available() else "cpu"
g_tfm_model = None
g_classifier_model = None
g_i3d_extractor = None
g_i3d_transforms = None # Sẽ chứa transform 1-crop

# --- Helper Class: I3DFeatureExtractor ---
class I3DFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        # model.blocks là một ModuleList
        
        # 1. Lấy các block 0-4 (phần ResNet body)
        # Chúng ta dùng nn.Sequential để chúng chạy tuần tự
        self.resnet_body = nn.Sequential(*model.blocks[:5])
        
        # 2. Lấy CHỈ lớp avg_pool từ block 5 (I3DHead)
        # model.blocks[5] là I3DHead
        self.avg_pool = model.blocks[5].avg_pool
            
    def forward(self, x):
        # x: [B, C, T, H, W] (ví dụ: [1, 3, 16, 224, 224])
        
        # 1. Chạy qua ResNet body
        # Input: [1, 3, 16, 224, 224]
        # Output: [1, 2048, 2, 7, 7]
        x = self.resnet_body(x)
        
        # 2. Chạy qua AvgPool
        # Input: [1, 2048, 2, 7, 7]
        # Output: [1, 2048, 1, 1, 1]
        x = self.avg_pool(x)
        
        # 3. Squeeze
        # Input: [1, 2048, 1, 1, 1]
        # Output: [2048] (vì B=1)
        return x.squeeze()

# --- Helper Function: Tải model (cho 1-Crop) ---
def load_all_models():
    """Tải 3 model (I3D, TFM, Classifier) lên RAM."""
    global g_tfm_model, g_classifier_model, g_i3d_extractor, g_i3d_transforms
    
    print(f"--- Đang tải models (sử dụng: {device}) ---")
    
    # 1. Tải I3D
    try:
        model_i3d = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
        model_i3d = model_i3d.to(device)
        model_i3d.eval()
        g_i3d_extractor = I3DFeatureExtractor(model_i3d)
        print("Tải I3D R50 thành công.")
    except Exception as e:
        print(f"LỖI khi tải I3D: {e}. Vui lòng BẬT INTERNET.")
        return False

    # 2. (MỚI) Tải I3D Transforms (1-Crop)
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    side_size = 256
    crop_size = 224 
    
    # (SỬA) Đây là transform 1-crop
    g_i3d_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(side_size),
        T.CenterCrop(crop_size), # <-- CHỈ LẤY CENTER CROP
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    print("Tải I3D (1-Crop) transforms thành công.")

    # 3. Tải TFM (Stage 1)
    try:
        g_tfm_model = RTFM_TFM().to(device) 
        g_tfm_model.load_state_dict(torch.load(TFM_MODEL_PATH, map_location=device))
        g_tfm_model.eval()
        print(f"Tải TFM model từ '{TFM_MODEL_PATH}' thành công.")
    except Exception as e:
        print(f"LỖI: Không tìm thấy TFM model tại: {TFM_MODEL_PATH} - Lỗi: {e}")
        return False

    # 4. Tải Classifier (Stage 2)
    try:
        g_classifier_model = Classifier().to(device)
        g_classifier_model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
        g_classifier_model.eval()
        print(f"Tải Classifier model từ '{CLASSIFIER_MODEL_PATH}' thành công.")
    except Exception as e:
        print(f"LỖI: Không tìm thấy Classifier model tại: {CLASSIFIER_MODEL_PATH} - Lỗi: {e}")
        return False
        
    print("--- Tải tất cả model thành công ---")
    return True

# --- Backend Function: Xử lý 1 clip (16 frame) (1-Crop) ---
def process_clip_1_crop(clip_frames_rgb):
    """
    Xử lý 1 clip (16 frame) và trả về 1 vector đặc trưng [2048]
    (Nhanh hơn 10-crop)
    """
    if len(clip_frames_rgb) < 16:
        return None
        
    # Áp dụng transform (1-crop) cho từng frame
    # (Kết quả là list các tensor [C, H_crop, W_crop])
    transformed_frames = [g_i3d_transforms(frame) for frame in clip_frames_rgb]
    
    # Stack [T, C, H, W] -> [C, T, H, W]
    base_tensor = torch.stack(transformed_frames).permute(1, 0, 2, 3) 
    
    # Thêm batch_dim (I3D cần 5D input)
    base_tensor = base_tensor.unsqueeze(0).to(device) # [1, C, T, H, W]

    with torch.no_grad():
        # (SỬA) Chỉ chạy 1 lần, không phải 10 lần
        features_avg = g_i3d_extractor(base_tensor) # [2048]
        
    return features_avg.cpu().numpy()

# --- Backend Function: Pad/Sample 32 snippets ---
def sample_pad_features(features_list):
    """
    Chuyển list N features thành 32 snippets cho RTFM
    """
    features_np = np.array(features_list)
    num_clips = features_np.shape[0]
    target_clips = 32
    
    print(f"Video có {num_clips} clip. Đang sample/pad về {target_clips} snippet...")
    
    if num_clips == 0:
        print("Lỗi: Video không trích xuất được clip nào.")
        return None
    
    if num_clips > target_clips:
        indices = np.random.choice(num_clips, target_clips, replace=False)
        indices.sort()
        features = features_np[indices]
    elif num_clips < target_clips:
        indices = np.arange(num_clips)
        indices = np.tile(indices, (target_clips // num_clips) + 1)[:target_clips]
        indices.sort()
        features = features_np[indices]
    else: # Bằng 32
        features = features_np
        
    # Thêm batch_dim và chuyển sang tensor
    return torch.from_numpy(features).float().unsqueeze(0).to(device) # [1, 32, C]

# --- Hàm Main ---
def main():
    # 1. Lấy đường dẫn video từ tham số
    if len(sys.argv) < 2:
        print("Sử dụng: python test_single_video.py \"<đường_dẫn_đến_video>\"")
        return
    
    video_path = sys.argv[1]
    if not Path(video_path).exists():
        print(f"Lỗi: Không tìm thấy file video tại: {video_path}")
        return

    # 2. Tải models
    if not load_all_models():
        print("Không thể tải model. Đang thoát...")
        return
        
    # 3. Mở video và trích xuất đặc trưng
    print(f"\n--- Bắt đầu trích xuất (1-Crop) cho: {video_path} ---")
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise Exception("Video rỗng hoặc không thể đọc.")
            
        frame_count = 0
        clip_frames_rgb = [] # Buffer 16 frame
        all_clip_features = [] # List các vector [2048]

        # Dùng tqdm để xem tiến trình
        with tqdm(total=total_frames, desc="Trích xuất (1-Crop)") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                pbar.update(1)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                clip_frames_rgb.append(frame_rgb)

                # Đủ 16 frame thì xử lý
                if len(clip_frames_rgb) == 16:
                    clip_feature = process_clip_1_crop(clip_frames_rgb)
                    if clip_feature is not None:
                        all_clip_features.append(clip_feature)
                    clip_frames_rgb = [] # Reset buffer

        cap.release()
        print(f"\nTrích xuất hoàn tất. Tổng cộng {len(all_clip_features)} clip.")
        
    except Exception as e:
        print(f"\nLỗi khi xử lý video: {e}")
        if cap: cap.release()
        return

    # 4. Pad/Sample về 32 snippets
    features_32 = sample_pad_features(all_clip_features)
    if features_32 is None:
        return
        
    # 5. Chạy qua 2 model RTFM
    print("Đang chạy model TFM và Classifier...")
    with torch.no_grad():
        robust_features = g_tfm_model(features_32) # [1, 32, C]
        robust_features_flat = robust_features.reshape(-1, 2048) # [32, C]
        predictions = g_classifier_model(robust_features_flat) # [32, 1]
    
    scores = predictions.cpu().numpy().squeeze()
    
    # 6. In kết quả
    print("\n" + "="*30)
    print("--- KẾT QUẢ 32 ĐIỂM BẤT THƯỜNG ---")
    print("="*30)
    
    for i, score in enumerate(scores):
        print(f"Snippet {i+1:02d} / 32: {score:.6f}")
        
    print("\n--- TÓM TẮT ---")
    print(f"Điểm cao nhất (Max Score): {np.max(scores):.6f}")
    print(f"Điểm trung bình (Avg Score): {np.mean(scores):.6f}")

if __name__ == "__main__":
    main()