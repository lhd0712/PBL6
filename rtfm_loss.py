# rtfm_loss.py
import torch
import torch.nn as nn

# Đây là hàm loss "ma thuật" của RTFM
def rtfm_loss_fn(normal_features, abnormal_features, margin=1.0):
    """
    normal_features: [Batch_N, Time, Channels]
    abnormal_features: [Batch_A, Time, Channels]
    """
    
    # 1. Tính độ lớn (Magnitude)
    # Shape: [Batch, Time]
    normal_mag = torch.norm(normal_features, p=2, dim=2)
    abnormal_mag = torch.norm(abnormal_features, p=2, dim=2)
    
    # 2. Ranking Loss (Loss chính)
    # Mục tiêu: Đẩy max(normal) > max(abnormal) + margin
    max_normal_mag = torch.max(normal_mag, dim=1)[0]   # [Batch_N]
    max_abnormal_mag = torch.max(abnormal_mag, dim=1)[0] # [Batch_A]
    
    # Dùng broadcasting [Batch_N, 1] - [1, Batch_A] -> [Batch_N, Batch_A]
    loss_rank = torch.clamp(
        margin - max_normal_mag.unsqueeze(1) + max_abnormal_mag.unsqueeze(0),
        min=0.0
    )
    loss_rank = loss_rank.mean()
    
    # 3. Sparsity Loss (Chỉ trên video Bất thường)
    # Mục tiêu: Chỉ một vài frame bất thường được có độ lớn cao
    # (Đẩy tổng các độ lớn thấp xuống)
    # Lấy top-k, giả sử k = 3
    k = 3 
    topk_abnormal_mag = torch.topk(abnormal_mag, k, dim=1)[0]
    loss_sparse = torch.mean(topk_abnormal_mag)
    
    # 4. Smoothness Loss (Chỉ trên video Bất thường)
    # Mục tiêu: Các frame liền kề không thay đổi độ lớn quá đột ngột
    diff = abnormal_mag[:, 1:] - abnormal_mag[:, :-1]
    loss_smooth = torch.mean(torch.pow(diff, 2))
    
    # Kết hợp (Tỷ lệ 8e-5 như trong code Keras/Sultani)
    LAMBDA = 8e-5
    total_loss = loss_rank + LAMBDA * (loss_sparse + loss_smooth)
    
    return total_loss