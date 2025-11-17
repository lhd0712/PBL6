# create_lists_stratified.py
import os
import glob
import random
from collections import defaultdict

# --- Cấu hình ---
FEATURE_ROOT = "I3D_Features"
GT_ROOT = "Ground_Truth_Files"
TRAIN_GT_FILE = os.path.join(GT_ROOT, "Anomaly_Train.txt")
TEST_GT_FILE = os.path.join(GT_ROOT, "Temporal_Anomaly_Annotation.txt")

OUTPUT_LIST_DIR = "feature_lists"
os.makedirs(OUTPUT_LIST_DIR, exist_ok=True)

VAL_RATIO = 0.2 
RANDOM_SEED = 42 

# --- ĐỊNH NGHĨA ID ---
CLASS_IDS = {
    "Normal": 0, "Abuse": 1, "Arrest": 2, "Arson": 3, "Assault": 4,
    "Burglary": 5, "Explosion": 6, "Fighting": 7, "RoadAccident": 8,
    "Robbery": 9, "Shooting": 10, "Shoplifting": 11, "Stealing": 12, "Vandalism": 13
}

# Đảo ngược để in tên class cho đẹp
ID_TO_NAME = {v: k for k, v in CLASS_IDS.items()}

def get_video_label(video_name):
    if "Normal_Videos" in video_name: return 0
    for name, cid in CLASS_IDS.items():
        if name == "Normal": continue
        if video_name.lower().startswith(name.lower()): return cid
    return -1

def write_list_file(video_list, output_path, feature_map):
    with open(output_path, 'w') as f:
        for video_name in video_list:
            if video_name in feature_map:
                path = os.path.abspath(feature_map[video_name]).replace('\\', '/')
                label = get_video_label(video_name)
                f.write(f"{path} {label}\n")

# --- 1. Quét features ---
print("Đang quét features...")
video_to_feature = {}
for path in glob.glob(os.path.join(FEATURE_ROOT, "**", "*.npy"), recursive=True):
    key = os.path.basename(path).replace('.npy', '')
    video_to_feature[key] = path

# --- 2. Stratified Split (Chia đều theo class) ---
print("\nĐang chia Train/Val theo tỷ lệ cân bằng (Stratified)...")

# Gom video theo nhãn: {0: [vid1, vid2], 1: [vid3], ...}
videos_by_class = defaultdict(list)

with open(TRAIN_GT_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        name = os.path.splitext(os.path.basename(line))[0]
        
        # Chỉ thêm nếu có feature
        if name not in video_to_feature:
            continue
            
        label = get_video_label(name)
        if label != -1:
            videos_by_class[label].append(name)

final_train_list = []
final_val_list = []

random.seed(RANDOM_SEED)

print(f"{'Class':<15} | {'Total':<5} | {'Train':<5} | {'Val':<5}")
print("-" * 40)

# Duyệt qua từng class để chia
for label, videos in videos_by_class.items():
    random.shuffle(videos) # Xáo trộn nội bộ class
    
    count = len(videos)
    val_count = int(count * VAL_RATIO)
    
    # Đảm bảo nếu có ít nhất 2 video thì val có 1, train có 1
    if count > 1 and val_count == 0:
        val_count = 1
        
    train_subset = videos[val_count:]
    val_subset = videos[:val_count]
    
    final_train_list.extend(train_subset)
    final_val_list.extend(val_subset)
    
    class_name = ID_TO_NAME.get(label, "Unknown")
    print(f"{class_name:<15} | {count:<5} | {len(train_subset):<5} | {len(val_subset):<5}")

# Xáo trộn lần cuối để không bị gom cụm khi train
random.shuffle(final_train_list)
random.shuffle(final_val_list)

# Ghi file
write_list_file(final_train_list, os.path.join(OUTPUT_LIST_DIR, "rtfm_train.list"), video_to_feature)
write_list_file(final_val_list, os.path.join(OUTPUT_LIST_DIR, "rtfm_val.list"), video_to_feature)

# --- 3. Test List (Giữ nguyên) ---
test_video_names = []
with open(TEST_GT_FILE, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if parts: test_video_names.append(parts[0].replace('.mp4', ''))
write_list_file(test_video_names, os.path.join(OUTPUT_LIST_DIR, "rtfm_test.list"), video_to_feature)

print(f"\n--- HOÀN TẤT ---")
print(f"Train: {len(final_train_list)} video")
print(f"Val:   {len(final_val_list)} video")