import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

class AudioDataset(Dataset):

    def __init__(self, root_dir, sr=16000, augment=False):

        self.files = []
        self.labels = []
        self.sr = sr
        self.augment = augment

        for label, folder in enumerate(sorted(os.listdir(root_dir))):
            folder_path = os.path.join(root_dir, folder)

            for file in os.listdir(folder_path):
                self.files.append(os.path.join(folder_path, file))
                self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        try:
            audio, _ = librosa.load(self.files[idx], sr=self.sr)
        except:
            return self.__getitem__((idx + 1) % len(self.files))

        max_length = self.sr * 10
        if self.augment:
            shift = np.random.randint(self.sr)
            audio = np.roll(audio, shift)

            noise = np.random.randn(len(audio))
            audio = audio + 0.003 * noise

            steps = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=steps)

            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)

        if len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)))
        else:
            audio = audio[:max_length]

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=128  #mel fgeayures
        )

        mel_db = librosa.power_to_db(mel)

        if self.augment:

            t = mel_db.shape[1]
            t_mask = np.random.randint(10, 30)
            t0 = np.random.randint(0, t - t_mask)
            mel_db[:, t0:t0+t_mask] = mel_db.min()

            f = mel_db.shape[0]
            f_mask = np.random.randint(5, 20)
            f0 = np.random.randint(0, f - f_mask)
            mel_db[f0:f0+f_mask, :] = mel_db.min()

        mel_db = torch.tensor(mel_db).unsqueeze(0).float()
        
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        
        return mel_db, self.labels[idx]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


path = 'src'
dataset = AudioDataset(path,augment=False)
indices = list(range(len(dataset)))

train_idx, test_idx = train_test_split(
    indices, 
    test_size=0.2,
    stratify = dataset.labels,
    random_state=42
)

train_dataset = Subset(AudioDataset(path, augment=True), train_idx)
test_dataset = Subset(AudioDataset(path, augment=False), test_idx)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)
x,y = train_dataset[0]



class CnnModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(1, 16, 3,padding =1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),

            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            # Block 2
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # Block 3
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)
        #flatten after conv layers
        x = torch.flatten(x,1)
        x = self.classifier(x)

        return x
import torch.nn as nn
import torch.optim as optim

model = CnnModel().to(device)  ## send my model to GPU

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

mylr = 3e-4
weight_d = 1e-4
optimizer = optim.AdamW(
    model.parameters(),
    lr=mylr,
    weight_decay=weight_d
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
)
from sklearn.metrics import f1_score
print(device)

def evaluate(model, loader):
    model.eval()
    all_pred = []
    all_label = []
    total = 0

    with torch.no_grad():
        for audio, label in loader: #validation loss
            audio = audio.to(device)
            label = label.to(device)

            output = model(audio)
            loss = criterion(output, label)

            pred = torch.argmax(output, dim=1)
            all_pred.extend(pred.detach().cpu().numpy())
            all_label.extend(label.detach().cpu().numpy())

            total += loss.item()

    f1 = f1_score(all_label, all_pred, average='macro') #mini
    total = total / len(loader) #val total loss

    return total, f1

epochs = 1
patience = 2

best_loss = float("inf")
patience_counter = 0

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for audio, label in train_loader:

        audio = audio.to(device)
        label = label.to(device)

        output = model(audio)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    scheduler.step(train_loss)

    current_lr = optimizer.param_groups[0]['lr']
    val_total, f1 = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"LR: {current_lr}")
    print(f"F1 Score of Val {f1}")
    print(f"Test Loss: {val_total}")

    if train_loss < best_loss:
        best_loss = train_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggerred")
        break

import torch
import torch.nn as nn

class BiLstmModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.bilstm = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers = 3,
            batch_first=True,  # ( B, W, H) Batch will at first
            bidirectional=True,
            dropout = 0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(256*2, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.squeeze(1) #to delete C in ( B, C, H, W)
        x = x.permute(0,2,1) # ( B, H, W )=> ( B, W, H)
        out1, out2 = self.bilstm(x) #output 1 tuple not 2 tupe ( ignoring out2 anyway )
        x = torch.mean(out1, dim = 1)
        x = self.classifier(x)

        return x

model_bilstm = BiLstmModel()

model2 = BiLstmModel().to(device)  ## send my model to GPU

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

mylr = 3e-4
weight_d = 1e-4
optimizer = optim.AdamW(
    model2.parameters(),
    lr=mylr,
    weight_decay=weight_d
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
)
from sklearn.metrics import f1_score
print(device)

def evaluate(model, loader):
    model.eval()
    all_pred = []
    all_label = []
    total = 0

    with torch.no_grad():
        for audio, label in loader: #validation loss
            audio = audio.to(device)
            label = label.to(device)

            output = model(audio)
            loss = criterion(output, label)

            pred = torch.argmax(output, dim=1)
            all_pred.extend(pred.detach().cpu().numpy())
            all_label.extend(label.detach().cpu().numpy())

            total += loss.item()

    f1 = f1_score(all_label, all_pred, average='macro') #mini
    total = total / len(loader) #val total loss

    return total, f1

epochs = 1
patience = 2

best_loss = float("inf")
patience_counter = 0

for epoch in range(epochs):

    model2.train()
    total_loss = 0

    for audio, label in train_loader:

        audio = audio.to(device)
        label = label.to(device)

        output = model2(audio)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #ecplode LSTM
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    scheduler.step(train_loss)

    current_lr = optimizer.param_groups[0]['lr']
    val_total, f1 = evaluate(model2, test_loader)
    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"LR: {current_lr}")
    print(f"F1 Score of Val {f1}")
    print(f"Test Loss: {val_total}")

    if val_total < best_loss:
        best_loss = val_total
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggerred")
        break

import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset

class TestAudioDataset(Dataset):

    def __init__(self, root_dir, sr=16000):

        self.files = []
        self.labels = []
        self.sr = sr

        for file in os.listdir(root_dir):
            self.files.append(os.path.join(root_dir, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        try:
            audio, _ = librosa.load(self.files[idx], sr=self.sr)
        except:
            return self.__getitem__((idx + 1) % len(self.files))

        max_length = self.sr * 10
        if len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)))
        else:
            audio = audio[:max_length]

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=128
        )
        mel_db = librosa.power_to_db(mel)
        mel_db = torch.tensor(mel_db).unsqueeze(0).float()
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        file_id = os.path.splitext(os.path.basename(self.files[idx]))[0]
        return mel_db, file_id
finaldataset= TestAudioDataset("src/testdataset")
finalloader = DataLoader(
    finaldataset, batch_size=32, shuffle=False
)

def final_test(model, loader):
    model.eval()
    all_pred = []
    all_label = []
    total = 0

    with torch.no_grad():
        for audio, file_name in loader: #validation loss
            audio = audio.to(device)
            output = model(audio)

            pred = torch.argmax(output, dim=1)
            all_pred.extend(pred.detach().cpu().numpy()) #Join the pred
            all_label.extend(file_name)
    return all_pred , all_label

all_pred, all_label = final_test(model, finalloader)
import pandas as pd
submission = pd.DataFrame({
    "id":all_label,
    "prediction":all_pred
})
submission.head(10)
id	prediction
0	0003	8
1	0010	4
2	0001	4
3	0004	1
4	0002	8
5	0009	8
6	0008	8
7	0006	8
8	0005	8
9	0007	8
maps = {
    0:"blues",
    1:"classical",
    2:"country",
    3:"disco",
    4:"hiphop",
    5:"jazz",
    6:"metal",
    7:"pop",
    8:"reggae",
    9:"rock"
}

submission['prediction'] = submission['prediction'].map(maps)
submission.head(10)
id	prediction
0	0003	reggae
1	0010	hiphop
2	0001	hiphop
3	0004	classical
4	0002	reggae
5	0009	reggae
6	0008	reggae
7	0006	reggae
8	0005	reggae
9	0007	reggae
submission.to_csv("submission.csv",index=False)
print("SUbmission Submitted")
SUbmission Submitted
C. Pretrained Model
1. Install & Load the HF
!pip install transformers -q
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

model = AutoModelForAudioClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2. Load Dataset for our HF MOdel
import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset

class AudioPreDataset(Dataset):

    def __init__(self, root_dir, sr=16000,feature_extractor=feature_extractor):

        self.files = []
        self.labels = []
        self.sr = sr
        self.feature_extractor = feature_extractor

        for label, folder in enumerate(sorted(os.listdir(root_dir))):
            folder_path = os.path.join(root_dir, folder)

            for file in os.listdir(folder_path):
                self.files.append(os.path.join(folder_path, file))
                self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        try:
            audio, _ = librosa.load(self.files[idx], sr=self.sr)
        except:
            return self.__getitem__((idx + 1) % len(self.files))

        max_length = self.sr * 5   # 10 seconds
        
        if len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)))
        else:
            audio = audio[:max_length]
            
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sr,
            return_tensors='pt'
        )
        input_values = inputs["input_values"].squeeze(0)
        return input_values, self.labels[idx]

pretraindataset = AudioPreDataset(
    "/kaggle/input/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/Data/genres_original",
    feature_extractor=feature_extractor,
)
from torch.utils.data import DataLoader
preloader = DataLoader(
    pretraindataset,
    batch_size=2,
    shuffle=False
)
import torch.nn as nn
import torch 
criterian = nn.CrossEntropyLoss()

optims = torch.optim.AdamW(model.parameters(), lr = 1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(1):
    model.train()
    total = 0
    for x, y in preloader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        logits = output.logits
        loss = criterian(logits, y)
        optims.zero_grad()
        loss.backward()
        optims.step()
        total += loss
    total = total/len(preloader)
    print(f"Epoch is {epoch +1} and loss is {total}")
/tmp/ipykernel_23/4114634376.py:29: UserWarning: PySoundFile failed. Trying audioread instead.
  audio, _ = librosa.load(self.files[idx], sr=self.sr)
/usr/local/lib/python3.12/dist-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
	Deprecated as of librosa version 0.10.0.
	It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)
Epoch is 1 and loss is 0.9307725429534912
path = './ast'

feature_extractor.save_pretrained(path)
model.save_pretrained(path)

class AudioDataset(Dataset):

    def __init__(self, root_dir, sr=16000, augment=False):

        self.files = []
        self.labels = []
        self.sr = sr
        self.augment = augment

        for label, folder in enumerate(sorted(os.listdir(root_dir))):
            folder_path = os.path.join(root_dir, folder)

            for file in os.listdir(folder_path):
                self.files.append(os.path.join(folder_path, file))
                self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        try:
            audio, _ = librosa.load(self.files[idx], sr=self.sr)
        except:
            return self.__getitem__((idx + 1) % len(self.files))

        max_length = self.sr * 10
        if self.augment:
            shift = np.random.randint(self.sr)
            audio = np.roll(audio, shift)

            noise = np.random.randn(len(audio))
            audio = audio + 0.003 * noise

            steps = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=steps)

            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)

        if len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)))
        else:
            audio = audio[:max_length]

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=128  #mel fgeayures
        )

        mel_db = librosa.power_to_db(mel)

        if self.augment:

            t = mel_db.shape[1]
            t_mask = np.random.randint(10, 30)
            t0 = np.random.randint(0, t - t_mask)
            mel_db[:, t0:t0+t_mask] = mel_db.min()

            f = mel_db.shape[0]
            f_mask = np.random.randint(5, 20)
            f0 = np.random.randint(0, f - f_mask)
            mel_db[f0:f0+f_mask, :] = mel_db.min()

        mel_db = torch.tensor(mel_db).unsqueeze(0).float()
        
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        
        return mel_db, self.labels[idx]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


path = '/kaggle/input/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/Data/genres_original'
dataset = AudioDataset(path,augment=False)
indices = list(range(len(dataset)))

train_idx, test_idx = train_test_split(
    indices, 
    test_size=0.2,
    stratify = dataset.labels,
    random_state=42
)

train_dataset = Subset(AudioDataset(path, augment=True), train_idx)
test_dataset = Subset(AudioDataset(path, augment=False), test_idx)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)
x,y = train_dataset[0]
print(x.shape)
torch.Size([1, 128, 313])
import torch
import torch.nn as nn

class CnnModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(1, 16, 3,padding =1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),

            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            # Block 2
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # Block 3
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)
        #flatten after conv layers
        x = torch.flatten(x,1)
        x = self.classifier(x)

        return x
import torch.nn as nn
import torch.optim as optim

model = CnnModel().to(device)  ## send my model to GPU

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

mylr = 3e-4
weight_d = 1e-4
optimizer = optim.AdamW(
    model.parameters(),
    lr=mylr,
    weight_decay=weight_d
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
)
from sklearn.metrics import f1_score
print(device)

def evaluate(model, loader):
    model.eval()
    all_pred = []
    all_label = []
    total = 0

    with torch.no_grad():
        for audio, label in loader: #validation loss
            audio = audio.to(device)
            label = label.to(device)

            output = model(audio)
            loss = criterion(output, label)

            pred = torch.argmax(output, dim=1)
            all_pred.extend(pred.detach().cpu().numpy())
            all_label.extend(label.detach().cpu().numpy())

            total += loss.item()

    f1 = f1_score(all_label, all_pred, average='macro') #mini
    total = total / len(loader) #val total loss

    return total, f1

epochs = 1
patience = 2

best_loss = float("inf")
patience_counter = 0

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for audio, label in train_loader:

        audio = audio.to(device)
        label = label.to(device)

        output = model(audio)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    scheduler.step(train_loss)

    current_lr = optimizer.param_groups[0]['lr']
    val_total, f1 = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"LR: {current_lr}")
    print(f"F1 Score of Val {f1}")
    print(f"Test Loss: {val_total}")

    if train_loss < best_loss:
        best_loss = train_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggerred")
        break
cuda
/tmp/ipykernel_23/2467175790.py:29: UserWarning: PySoundFile failed. Trying audioread instead.
  audio, _ = librosa.load(self.files[idx], sr=self.sr)
/usr/local/lib/python3.12/dist-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
	Deprecated as of librosa version 0.10.0.
	It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)
Epoch 1
Train Loss: 2.2885
LR: 0.0003
F1 Score of Val 0.0360836961780358
Test Loss: 2.287916762488229
B. Bi-GRU MODEL
import torch
import torch.nn as nn

class BiLstmModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.bilstm = nn.GRU(
            input_size=128,
            hidden_size=256,
            num_layers = 3,
            batch_first=True,  # ( B, W, H) Batch will at first
            bidirectional=True,
            dropout = 0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(256*2, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.squeeze(1) #to delete C in ( B, C, H, W)
        x = x.permute(0,2,1) # ( B, H, W )=> ( B, W, H)
        out1, out2 = self.bilstm(x) #output 1 tuple not 2 tupe ( ignoring out2 anyway )
        x = torch.mean(out1, dim = 1)
        x = self.classifier(x)

        return x

model_bilstm = BiLstmModel()
print(model_bilstm)
BiLstmModel(
  (bilstm): GRU(128, 256, num_layers=3, batch_first=True, dropout=0.3, bidirectional=True)
  (classifier): Sequential(
    (0): Linear(in_features=512, out_features=128, bias=True)
    (1): LeakyReLU(negative_slope=0.1)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=128, out_features=10, bias=True)
  )
)
import torch.nn as nn
import torch.optim as optim

model2 = BiLstmModel().to(device)  ## send my model to GPU

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

mylr = 3e-4
weight_d = 1e-4
optimizer = optim.AdamW(
    model2.parameters(),
    lr=mylr,
    weight_decay=weight_d
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
)
from sklearn.metrics import f1_score
print(device)

def evaluate(model, loader):
    model.eval()
    all_pred = []
    all_label = []
    total = 0

    with torch.no_grad():
        for audio, label in loader: #validation loss
            audio = audio.to(device)
            label = label.to(device)

            output = model(audio)
            loss = criterion(output, label)

            pred = torch.argmax(output, dim=1)
            all_pred.extend(pred.detach().cpu().numpy())
            all_label.extend(label.detach().cpu().numpy())

            total += loss.item()

    f1 = f1_score(all_label, all_pred, average='macro') #mini
    total = total / len(loader) #val total loss

    return total, f1

epochs = 1
patience = 2

best_loss = float("inf")
patience_counter = 0

for epoch in range(epochs):

    model2.train()
    total_loss = 0

    for audio, label in train_loader:

        audio = audio.to(device)
        label = label.to(device)

        output = model2(audio)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #ecplode LSTM
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    scheduler.step(train_loss)

    current_lr = optimizer.param_groups[0]['lr']
    val_total, f1 = evaluate(model2, test_loader)
    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"LR: {current_lr}")
    print(f"F1 Score of Val {f1}")
    print(f"Test Loss: {val_total}")

    if val_total < best_loss:
        best_loss = val_total
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggerred")
        break
cuda
/tmp/ipykernel_23/2467175790.py:29: UserWarning: PySoundFile failed. Trying audioread instead.
  audio, _ = librosa.load(self.files[idx], sr=self.sr)
/usr/local/lib/python3.12/dist-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
	Deprecated as of librosa version 0.10.0.
	It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)
Epoch 1
Train Loss: 2.3098
LR: 0.0003
F1 Score of Val 0.036571428571428574
Test Loss: 2.2827654906681607
Save Cnn and Bi-LSTM Model
torch.save(model2.state_dict(), "bi-lstm_model.pt")
Evaluation
import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset

class TestAudioDataset(Dataset):

    def __init__(self, root_dir, sr=16000):

        self.files = []
        self.labels = []
        self.sr = sr

        for file in os.listdir(root_dir):
            self.files.append(os.path.join(root_dir, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        try:
            audio, _ = librosa.load(self.files[idx], sr=self.sr)
        except:
            return self.__getitem__((idx + 1) % len(self.files))

        max_length = self.sr * 10
        if len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)))
        else:
            audio = audio[:max_length]

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=128
        )
        mel_db = librosa.power_to_db(mel)
        mel_db = torch.tensor(mel_db).unsqueeze(0).float()
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        file_id = os.path.splitext(os.path.basename(self.files[idx]))[0]
        return mel_db, file_id
finaldataset= TestAudioDataset("/kaggle/input/datasets/samasiayushman/testdataset")
finalloader = DataLoader(
    finaldataset, batch_size=32, shuffle=False
)
for x, y in finalloader :
    print(x)
    print(y)
    break
tensor([[[[ 6.3580e-01,  8.4151e-02, -3.6836e-01,  ...,  1.0511e+00,
            3.7858e-01,  1.3159e+00],
          [ 2.1757e+00,  2.4375e+00,  1.9794e+00,  ...,  2.7811e+00,
            2.5190e+00,  2.4832e+00],
          [ 2.3626e+00,  2.6153e+00,  2.0855e+00,  ...,  2.5942e+00,
            2.5456e+00,  2.3864e+00],
          ...,
          [ 6.3944e-01,  8.3403e-01,  2.7940e-01,  ...,  3.9747e-01,
           -3.5140e-01, -1.0621e+00],
          [ 2.3530e-02,  1.9047e-01, -4.4673e-01,  ..., -3.7094e-01,
           -8.5109e-01, -1.4202e+00],
          [-1.5133e+00, -1.2468e+00, -1.5892e+00,  ..., -2.0369e+00,
           -2.4750e+00, -2.6448e+00]]],


        [[[-7.3220e-01, -4.1535e-01, -4.1809e-01,  ..., -1.0253e+00,
           -1.1612e+00, -1.2104e+00],
          [-8.2219e-02, -6.9375e-02,  2.8998e-01,  ..., -5.0618e-01,
           -3.7047e-01, -4.6551e-01],
          [ 5.2995e-01,  5.0037e-01,  3.1132e-01,  ..., -6.3819e-01,
           -3.0674e-01, -1.9558e-01],
          ...,
          [-1.5549e+00, -1.3900e+00, -8.3753e-01,  ..., -2.7366e+00,
           -2.8182e+00, -3.1513e+00],
          [-1.9495e+00, -1.7024e+00, -1.1809e+00,  ..., -3.0714e+00,
           -2.9758e+00, -3.3980e+00],
          [-2.9091e+00, -2.8159e+00, -2.4420e+00,  ..., -4.3201e+00,
           -4.2828e+00, -4.3093e+00]]],


        [[[ 4.5405e-01,  7.2027e-02, -8.0136e-01,  ..., -3.0250e-01,
           -3.4008e-01, -5.6613e-02],
          [ 5.1626e-01,  2.1839e-01, -4.4956e-01,  ..., -2.1654e-01,
           -2.2996e-01,  2.0669e-01],
          [ 5.2648e-01,  1.5246e-01, -2.9198e-01,  ...,  1.8199e-01,
            1.6058e-01,  2.0759e-01],
          ...,
          [-1.4044e+00, -1.1965e+00, -1.1387e+00,  ..., -8.3639e-01,
           -1.0276e+00, -1.2863e+00],
          [-1.5889e+00, -1.2363e+00, -1.1357e+00,  ..., -7.0677e-01,
           -1.0674e+00, -1.2162e+00],
          [-1.8407e+00, -1.8407e+00, -1.8407e+00,  ..., -1.3752e+00,
           -1.6622e+00, -1.8258e+00]]],


        ...,


        [[[-3.3929e-02, -3.4585e-01, -5.0056e-01,  ..., -5.4198e-01,
           -6.5284e-01, -4.2986e-03],
          [ 1.9586e-01, -6.9398e-02, -3.9802e-01,  ..., -1.3278e-01,
           -1.0100e-01,  1.3638e-01],
          [ 4.8065e-01,  6.3182e-01,  5.9078e-01,  ..., -9.1070e-02,
            3.7954e-03,  4.3782e-01],
          ...,
          [-1.8817e+00, -1.8817e+00, -1.8445e+00,  ..., -1.8817e+00,
           -1.8817e+00, -1.8817e+00],
          [-1.8817e+00, -1.7387e+00, -1.5987e+00,  ..., -1.8817e+00,
           -1.8817e+00, -1.8817e+00],
          [-1.8817e+00, -1.8817e+00, -1.8817e+00,  ..., -1.8817e+00,
           -1.8817e+00, -1.8817e+00]]],


        [[[ 6.3030e-01,  1.8185e-01, -6.9288e-01,  ..., -1.1026e+00,
           -1.0836e+00, -5.5162e-02],
          [ 1.3300e+00,  1.2144e+00,  9.5135e-01,  ...,  9.6095e-01,
            1.1142e+00,  1.0388e+00],
          [ 1.8871e+00,  2.1922e+00,  2.2292e+00,  ...,  2.1303e+00,
            2.0296e+00,  1.6530e+00],
          ...,
          [-8.3440e-01, -9.1979e-01, -1.6380e+00,  ..., -1.8077e+00,
           -1.3785e+00, -1.2447e+00],
          [-9.6405e-01, -1.0486e+00, -1.8325e+00,  ..., -2.0056e+00,
           -1.5341e+00, -1.3400e+00],
          [-1.7646e+00, -1.9733e+00, -2.5745e+00,  ..., -2.5745e+00,
           -2.5745e+00, -1.8074e+00]]],


        [[[-2.7475e+00, -2.2418e+00, -1.8582e+00,  ..., -3.6368e-01,
            3.7831e-02, -2.1943e-01],
          [-1.9856e+00, -1.4063e+00, -8.1992e-01,  ...,  1.7868e+00,
            2.2665e+00,  1.7247e+00],
          [-1.4109e+00, -1.0822e+00, -6.1556e-01,  ...,  2.6792e+00,
            2.8756e+00,  1.9217e+00],
          ...,
          [-1.9478e+00, -1.7951e+00, -1.8589e+00,  ..., -1.7074e+00,
           -1.9796e+00, -2.2095e+00],
          [-2.2691e+00, -2.0408e+00, -2.1601e+00,  ..., -2.3143e+00,
           -2.3426e+00, -2.3792e+00],
          [-3.7008e+00, -3.5679e+00, -3.4658e+00,  ..., -3.7802e+00,
           -3.8287e+00, -4.1552e+00]]]])
('0003', '0010', '0001', '0004', '0002', '0009', '0008', '0006', '0005', '0007')
def final_test(model, loader):
    model.eval()
    all_pred = []
    all_label = []
    total = 0

    with torch.no_grad():
        for audio, file_name in loader: #validation loss
            audio = audio.to(device)
            output = model(audio)

            pred = torch.argmax(output, dim=1)
            all_pred.extend(pred.detach().cpu().numpy()) #Join the pred
            all_label.extend(file_name)
    return all_pred , all_label

all_pred, all_label = final_test(model, finalloader)
import pandas as pd
submission = pd.DataFrame({
    "id":all_label,
    "prediction":all_pred
})
submission.head(10)
id	prediction
0	0003	8
1	0010	4
2	0001	4
3	0004	1
4	0002	8
5	0009	8
6	0008	8
7	0006	8
8	0005	8
9	0007	8
maps = {
    0:"blues",
    1:"classical",
    2:"country",
    3:"disco",
    4:"hiphop",
    5:"jazz",
    6:"metal",
    7:"pop",
    8:"reggae",
    9:"rock"
}

submission['prediction'] = submission['prediction'].map(maps)
submission.head(10)
id	prediction
0	0003	reggae
1	0010	hiphop
2	0001	hiphop
3	0004	classical
4	0002	reggae
5	0009	reggae
6	0008	reggae
7	0006	reggae
8	0005	reggae
9	0007	reggae
submission.to_csv("submission.csv",index=False)
print("SUbmission Submitted")
SUbmission Submitted
C. Pretrained Model
1. Install & Load the HF
!pip install transformers -q
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

model = AutoModelForAudioClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2. Load Dataset for our HF MOdel
import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset

class AudioPreDataset(Dataset):

    def __init__(self, root_dir, sr=16000,feature_extractor=feature_extractor):

        self.files = []
        self.labels = []
        self.sr = sr
        self.feature_extractor = feature_extractor

        for label, folder in enumerate(sorted(os.listdir(root_dir))):
            folder_path = os.path.join(root_dir, folder)

            for file in os.listdir(folder_path):
                self.files.append(os.path.join(folder_path, file))
                self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        try:
            audio, _ = librosa.load(self.files[idx], sr=self.sr)
        except:
            return self.__getitem__((idx + 1) % len(self.files))

        max_length = self.sr * 5   # 10 seconds
        
        if len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)))
        else:
            audio = audio[:max_length]
            
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sr,
            return_tensors='pt'
        )
        input_values = inputs["input_values"].squeeze(0)
        return input_values, self.labels[idx]

pretraindataset = AudioPreDataset(
    "/kaggle/input/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/Data/genres_original",
    feature_extractor=feature_extractor,
)
from torch.utils.data import DataLoader
preloader = DataLoader(
    pretraindataset,
    batch_size=2,
    shuffle=False
)
import torch.nn as nn
import torch 
criterian = nn.CrossEntropyLoss()

optims = torch.optim.AdamW(model.parameters(), lr = 1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(1):
    model.train()
    total = 0
    for x, y in preloader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        logits = output.logits
        loss = criterian(logits, y)
        optims.zero_grad()
        loss.backward()
        optims.step()
        total += loss
    total = total/len(preloader)
    print(f"Epoch is {epoch +1} and loss is {total}")
/tmp/ipykernel_23/4114634376.py:29: UserWarning: PySoundFile failed. Trying audioread instead.
  audio, _ = librosa.load(self.files[idx], sr=self.sr)
/usr/local/lib/python3.12/dist-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
	Deprecated as of librosa version 0.10.0.
	It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)
Epoch is 1 and loss is 0.9307725429534912
path = './ast'

feature_extractor.save_pretrained(path)
model.save_pretrained(path)
