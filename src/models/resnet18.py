import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib.pyplot import plt

device = torch.device('mps' if torch.mps.is_availabel() else 'cpu')

# 하이퍼파라미터 설정
batch_size = 64
num_epochs = 10
learning_rate = 0.0001

# 데이터 불러오기 & 로드
transform = transforms.compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.MPS(root='/Users/iujeong/04_luxating_patella/dataset', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MPS(root='/Users/iujeong/04_luxating_patella/dataset', train=False, transform=transform, download=True)

train_loader = torch.utils.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 모델 정의
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        if stride != 1 or in_channel != out_channel:
            self.proj = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)
            self.bn_proj = nn.BatchNorm2d(out_channel)

        else:
            self.proj = None

        self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.proj is not None:
            identity = self.bn_proj(self.proj(x))

        out += identity

        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, padding=1, kernel_size=3)
        self.bn = nn.BatchNorm2d(64)

        self.resblock1_1 = ResBlock(64, 64, stride=1)
        self.resblock1_2 = ResBlock(64, 64, stride=1)

        self.resblock2_1 = ResBlock(64, 128, stride=2)
        self.resblock2_2 = ResBlock(128, 128, stride=1)

        self.resblock3_1 = ResBlock(128, 256, stride=2)
        self.resblock3_2 = ResBlock(256, 256, stride=1)

        self.resblock4_1 = ResBlock(256, 512, stride=2)
        self.resblock4_2 = ResBlock(512, 512, stride=1)

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*1*1, 10)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.resblock1_1(out)
        out = self.resblock1_2(out)

        out = self.resblock2_1(out)
        out = self.resblock2_2(out)

        out = self.resblock3_1(out)
        out = self.resblock3_2(out)

        out = self.resblock4_1(out)
        out = self.resblock4_2(out)

        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

model = ResNet().to(device)

# optimizer, loss 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 반복문 (학습)
train_loss_list = []
test_loss_list = []

train_acc_list = []
test_acc_list = []

train_step = len(train_loader)
test_step = len(test_loader)

for epoch in range(num_epochs):
    n_samples = 0
    epoch_loss = 0
    correct = 0

    model.train()
    for i, (image, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        n_samples += labels.size(0)

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], step [{i+1}/{train_step}], train loss [{correct/n_samples*100:.2f}]')

    train_loss_list.append(epoch_loss/train_step)
    train_acc_list.append(correct/n_samples*100)

    with torch.no_grad():
        n_samples = 0
        epoch_loss = 0
        correct = 0

        model.eval()

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            n_samples += labels.size(0)
            epoch_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

        print(f'test epoch [{epoch+1}/{num_epochs}], test loss [{epoch_loss/test_step:.4f}], test accuracy [{correct/n_samples*100:.2f}]')

        test_loss_list.append(epoch_loss/test_step)
        test_acc_list.append(correct/n_samples*100)



plt.figure()
x_epochs = list( range(1,num_epochs+1) )
plt.plot(x_epochs, train_loss_list, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid()
plt.show()
plt.clf()

plt.figure()
x_epochs = list( range(1,num_epochs+1) )
plt.plot(x_epochs, train_acc_list, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()
plt.grid()
plt.show()
plt.clf()