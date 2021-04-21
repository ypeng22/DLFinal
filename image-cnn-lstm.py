import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
subjects = ['S01', 'S02', 'S03', 'S04', 'S05','S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12']
actions = ['A01', 'A02', 'A03', 'A04', 'A05','A06', 'A07', 'A08', 'A09', 'A10', 'A11']
reps = ['R01', 'R02', 'R03', 'R04', 'R05']

class BerkeleyMHAD(Dataset):

    def __init__(self, vid_names, root_dir, classes, transform=None):
        self.vid_names = vid_names # list of file names for videos (ex. S01_A01_R01)
        self.root_dir = root_dir # directory where videos are stored
        self.transform = transform
        self.classes = classes
        
    def __len__(self):
        return len(self.vid_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = os.path.join(self.root_dir, self.vid_names[idx])
        x = np.load(path)['x']
        label = self.classes.index(np.load(path)['y']) #my goals are beyond your understanding
        sample = {'x': x, 'y': label}

        if self.transform:
            sample = self.transform(sample)
        return sample
    
class cnn_lstm(nn.Module):
    def __init__(self, classes):
        super(cnn_lstm, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5,)
        self.conv2 = nn.Conv2d(3, 3, 5)
        self.pool1 = nn.MaxPool2d(3)
        self.n_hidden = 100
        self.n_layers = 1
        self.l_lstm = torch.nn.LSTM(input_size = 1296, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, classes)
        self.relu = nn.LeakyReLU(.1)
        self.soft = nn.Softmax(dim = 0)
    def forward(self, x):
        batch = x.shape[0]
        #intialize lstm hidden state
        hidden_state = torch.zeros(self.n_layers, 1, self.n_hidden).to(dev)
        cell_state = torch.zeros(self.n_layers, 1, self.n_hidden).to(dev)
        self.hidden = (hidden_state, cell_state)
        
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool1(self.relu(self.conv2(x)))
        x = x.reshape(batch, -1).unsqueeze(0)
        #print(x.shape)
        lstm_out, _ = self.l_lstm(x, self.hidden) #lstm_out shape is batch_size, seq len, hidden state
        #print(lstm_out.shape)
        lstm_out = lstm_out[:,-1,:]
        #print(lstm_out.shape)
        lstm_out = self.relu(self.fc1(lstm_out.squeeze()))
        lstm_out = self.relu(self.fc2(lstm_out))
        lstm_out = self.soft(lstm_out)
        return lstm_out
    
def check(i):
    #insert more i.find terms for each action
    return i.find('A01') != -1 or i.find('A07') != -1
'''vid_names = [i for i in next(os.walk('./rgb_video_data'))[2] if check(i)]
train_vid_names = [i for i in vid_names if i.find('S09') == -1 and i.find('S10') == -1 and i.find('S11') == -1 and i.find('S12') == -1]
valid_vid_names = [i for i in vid_names if i.find('S09') != -1 ]
test_vid_names = [i for i in vid_names if i.find('S10') != -1 or i.find('S11') != -1 or i.find('S12') != -1]

#data shape is (num_pics, height, width, channel)
train_dataset = BerkeleyMHAD(train_vid_names, './rgb_video_data', classes = [0, 6])
valid_dataset = BerkeleyMHAD(valid_vid_names, './rgb_video_data', classes = [0, 6])
test_dataset = BerkeleyMHAD(test_vid_names, './rgb_video_data', classes = [0, 6])
plt.imshow(train_dataset[0]['x'][0])'''

batch_size = 1

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#change 2 to number of classes
model = cnn_lstm(2).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=.01)
criterion = nn.CrossEntropyLoss()
epoch = 1
#train

tguess = []
tanswers = []
tcorrect = 0
ttotal = 0
for t in range(epoch):
    train_loss = 0
    valid_loss = 0
    tguess = []
    tanswers = []
    tcorrect = 0
    ttotal = 0
    for i in range(len(train_dataset)):
        data = train_dataset[i]
        inpt = torch.tensor(data['x'], dtype=torch.float).permute(0, 3, 1, 2).to(dev)
        label = torch.tensor(data['y']).unsqueeze(0).to(dev)
        output = model(inpt).unsqueeze(0)
        loss = criterion(output, label) #.view(-1)
        loss.backward()
        optimizer.step()  
        optimizer.zero_grad()
        train_loss += loss.item()
        
        if torch.argmax(output.squeeze()) == label:
            tcorrect += 1
        ttotal += 1
        tanswers.append(label.item())
        tguess.append(output[0][1].item())
        
    torch.cuda.empty_cache()   

    with torch.no_grad():
        for i in range(len(valid_dataset)):
            data = valid_dataset[i]
            inpt = torch.tensor(data['x'], dtype=torch.float).permute(0, 3, 1, 2).to(dev)
            label = torch.tensor(data['y']).unsqueeze(0).to(dev)
            output = model(inpt).unsqueeze(0)
            loss = criterion(output, label)
            valid_loss += loss.item()
    print("epoch:", valid_loss / len(valid_dataset), train_loss / len(train_dataset))


guess = []
answers = []
correct = 0
total = 0
with torch.no_grad():
    for i in range(len(test_dataset)):
        data = test_dataset[i]
        inpt = torch.tensor(data['x'], dtype=torch.float).permute(0, 3, 1, 2).to(dev)
        label = torch.tensor(data['y']).to(dev)
        output2 = model(inpt)
        if torch.argmax(output2.squeeze()) == label:
            correct += 1
        total += 1
        answers.append(label.item())
        guess.append(output2[1].item())
#guess = torch.argmax(guess.squeeze(), dim=1)
#guess = np.array(guess).squeeze()
print(correct / total)
#print('CNN AUC: %.4f' % roc_auc_score(testlabel, guess), ' AUPRC: %.4f' % average_precision_score(testlabel, guess))
torch.save(model, 'cnn_lstm.torch')
