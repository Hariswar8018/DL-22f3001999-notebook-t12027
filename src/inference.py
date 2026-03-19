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

model = BiLstmModel()
model.load_state_dict(torch.load("models/cnn_model.pt"))

import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
model.to(device)
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
