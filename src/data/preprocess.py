import librosa
import pandas as pd 
import numpy as np 
import librosa
import os 


split_file="data/raw/ICBHI_challenge_train_test.txt"
DATA_DIR="data/raw/ICBHI_final_database"
TARGET_SR=16000

df=pd.read_csv(split_file,sep="\t",names=['filename','set_type'])
TARGET_DURATION=8
TARGET_SAMPLES=TARGET_DURATION * TARGET_SR
DEVICE_MAP = {
    'AKGC417L': 0,
    'LittC2SE': 1,
    'Litt3200': 2,
    'Meditron': 3
}
def cyclic_padding(wav, target_len):
    """Repeat signal cyclically until target length is reached."""
    if len(wav) >= target_len:
        return wav[:target_len]
    repeat_count = (target_len // len(wav)) + 1
    padded = np.tile(wav, repeat_count)
    return padded[:target_len]
# Storage lists
X_train, y_train, device_train = [], [], []
X_test, y_test, device_test = [], [], []

print(df.head())
for index,row in df.iterrows():
    wav_path=os.path.join(DATA_DIR,row['filename']+".wav")
    txt_path=os.path.join(DATA_DIR,row['filename']+".txt")
    print(wav_path,txt_path)
    if not os.path.exists(wav_path) or not os.path.exists(txt_path):
        print(f"File not found {wav_path} {txt_path}")
        continue
    audio,_=librosa.load(wav_path,sr= TARGET_SR)
    anns=pd.read_csv(txt_path,sep="\t",names=["start","end","crackle","wheeze"])  
    dev_name = row['filename'].split('_')[-1]
    dev_id = DEVICE_MAP.get(dev_name, -1)
    for _, ann in anns.iterrows():
        start = int(ann['start'] * TARGET_SR)
        end = int(ann['end'] * TARGET_SR)
        chunk = audio[start:end]
        if len(chunk) < 100:  # skip tiny chunks
            continue

        processed_wav = cyclic_padding(chunk, TARGET_SAMPLES)

         # Step 6: Assign label
        c, w = ann['crackle'], ann['wheeze']
        if c == 0 and w == 0:
            label = 0  # Normal
        elif c == 1 and w == 0:
            label = 1  # Crackle
        elif c == 0 and w == 1:
            label = 2  # Wheeze
        else:
            label = 3  # Both
        # Append to train or test
        if row['set_type'] == 'train':
            X_train.append(processed_wav)
            y_train.append(label)
            device_train.append(dev_id)
        else:
            X_test.append(processed_wav)
            y_test.append(label)
            device_test.append(dev_id)
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)
device_train = np.array(device_train, dtype=np.int64)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int64)
device_test = np.array(device_test, dtype=np.int64)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
# Save
OUTPUT_PATH = "data/processed/icbhi_ast_16k_8s.npz"
np.savez(OUTPUT_PATH,
         X_train=X_train, y_train=y_train, device_train=device_train,
         X_test=X_test, y_test=y_test, device_test=device_test)
print(f"✅ Saved: {OUTPUT_PATH}")