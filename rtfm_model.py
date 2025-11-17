# rtfm_model.py
import torch
import torch.nn as nn

# Đây là mô hình chính của RTFM
class RTFM_TFM(nn.Module):
    def __init__(self, input_dim=2048, num_layers=3, hidden_dim=2048):
        super(RTFM_TFM, self).__init__()
        
        layers = []
        for i in range(num_layers):
            # Mỗi lớp là 1D-Conv với kernel=3
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        # Input x: [Batch, Time, Channels] (ví dụ: [B, 32, 2048])
        # Conv1D cần: [Batch, Channels, Time]
        x = x.permute(0, 2, 1) 
        
        out = self.feature_extractor(x)
        
        # Chuyển về: [Batch, Time, Channels]
        return out.permute(0, 2, 1)

# Đây là mô hình phụ, dùng ở Giai đoạn 2 (giống hệt Sultani)
class Classifier(nn.Module):
    def __init__(self, input_dim=2048, num_classes=14): # Mặc định 14 lớp
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.6)
        
        self.fc2 = nn.Linear(512, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.6)
        
        # (SỬA) Output ra num_classes (14) thay vì 1
        self.fc3 = nn.Linear(32, num_classes)
        
        # (SỬA) Bỏ Sigmoid cuối cùng

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        
        # Output là Logits (điểm thô), chưa qua Softmax
        x = self.fc3(x) 
        return x