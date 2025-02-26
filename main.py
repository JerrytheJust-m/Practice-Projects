import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from tqdm import tqdm


class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data = ImageFolder(data_dir, transform = transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def classes(self):
        return self.data.classes


class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


data_dir='archive/train'
train_folder = data_dir
valid_folder = 'archive/valid'
test_folder = 'archive/test'

#dataset = PlayingCardDataset(data_dir=data_dir)
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
#dataset = PlayingCardDataset(data_dir=data_dir, transform=transform)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

train_dataset = PlayingCardDataset(train_folder, transform=transform)
valid_dataset = PlayingCardDataset(valid_folder, transform=transform)
test_dataset = PlayingCardDataset(valid_folder, transform=transform)


batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

#creat training loop
num_epoch = 3
train_losses, valid_losses = [], []
model = SimpleCardClassifier(num_classes=53)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

#change to GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

if __name__ == "__main__":
    print(device)
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc='Training loop'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc='Validation loop'):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        valid_loss = running_loss / len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        print(epoch, train_loss, valid_loss)
    torch.save(model.state_dict(), "PlayCard.pth")