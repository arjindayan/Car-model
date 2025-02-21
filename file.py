import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam



import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets



import os
import shutil
import random
from tqdm.auto import tqdm
from helper_functions import accuracy_fn
import matplotlib.pyplot as plt


device="cuda" if torch.cuda.is_available() else "cpu"



train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  # Rastgele döndürme
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Renk değişimleri
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Rastgele kaydırma
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet ortalaması
                         [0.229, 0.224, 0.225])
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root="dataset\\train", transform=train_transform)
test_dataset  = datasets.ImageFolder(root="dataset\\test", transform=test_transform)

train_dataloader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
test_dataloader=DataLoader(dataset=test_dataset,batch_size=32,shuffle=False)


print("Class to index mapping:", train_dataset.class_to_idx)
print("Classes:", train_dataset.classes)



def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               scheduler,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.train()
    
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_lambda * l2_norm
        
        train_loss += loss.item()
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    
    scheduler.step(train_loss)
    
    return train_loss, train_acc

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.eval()
    
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            
            test_loss += loss.item()
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        
    return test_loss, test_acc

class model(nn.Module):
    def __init__(self, in_shape, hidden_shape, out_shape):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=hidden_shape, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(hidden_shape),  # Batch Normalization eklendi
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape, out_channels=hidden_shape*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_shape*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)  # Dropout eklendi
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_shape*2, out_channels=hidden_shape*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_shape*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape*2, out_channels=hidden_shape*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_shape*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        self.block3 = nn.Sequential(  # Yeni blok eklendi
            nn.Conv2d(in_channels=hidden_shape*4, out_channels=hidden_shape*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_shape*4),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape*4, out_channels=hidden_shape*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_shape*8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_shape*8*28*28, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=out_shape)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

# EarlyStopping sınıfını kodun başına taşı (import'lardan sonra)
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# Modeli device'a taşı ve önceki ağırlıkları yükle
sample_model = model(in_shape=3, hidden_shape=32, out_shape=3).to(device)
sample_model.load_state_dict(torch.load('best_model.pth'))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=sample_model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.1, 
    patience=3, 
    verbose=True
)

# Hiperparametreler
EPOCHS = 70
PATIENCE = 7

# Early stopping - aynı dosyaya kaydetmek için
early_stopping = EarlyStopping(patience=PATIENCE, path='best_model.pth')

# Eğitim geçmişi için listeler
train_losses = []
test_losses = []
train_accs = []
test_accs = []

# Tek bir eğitim döngüsü
for epoch in range(EPOCHS):
    print(f"\nEpoch: {epoch+1}/{EPOCHS}")
    print("-" * 30)
    
    train_loss, train_acc = train_step(
        model=sample_model,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        scheduler=scheduler,
        device=device
    )
    
    test_loss, test_acc = test_step(
        data_loader=test_dataloader,
        model=sample_model,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
    
    early_stopping(test_loss, sample_model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

def plot_training_history(train_losses, test_losses, train_accs, test_accs):
    plt.figure(figsize=(12, 4))
    
    # Loss grafiği
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy grafiği
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Eğitim sonrası görselleştirme
plot_training_history(train_losses, test_losses, train_accs, test_accs)



















































# # Orijinal klasörlerin yolu (her bir klasör içinde JPEG dosyaları var)
# base_folder = 'original_dataset'  # örn: 'original_dataset/ustten', 'original_dataset/ayni_hiza', 'original_dataset/alttan'

# # Yeni veri setini koymak istediğiniz yol
# output_folder = 'dataset'  # Sonra dataset/train, dataset/val, dataset/test gibi alt klasörler olacak

# # Sınıflarınız (klasör isimleri)
# classes = ['above', 'same_level', 'below']

# # Oranlar (train, val, test)
# train_ratio = 0.75
# test_ratio = 0.25

# for cls in classes:
#     # Orijinal sınıf klasörünü oku
#     class_folder = os.path.join(base_folder, cls)
#     images = os.listdir(class_folder)
#     images = [img for img in images if img.endswith('.jpg') or img.endswith('.jpeg') or img.endswith('.png')]
    
#     # Karıştır
#     random.shuffle(images)
    
#     # Kaç tane resim olduğu
#     total_images = len(images)
#     train_count = int(total_images * train_ratio)
#     test_count = total_images - train_count  # geri kalan
    
#     # Train/Val/Test için dosyaları ayır
#     train_images = images[:train_count]
#     test_images = images[train_count:]
    
#     # Klasörleri oluştur (örneğin dataset/train/ustten, dataset/val/ustten, dataset/test/ustten)
#     for split in ['train', 'test']:
#         split_folder = os.path.join(output_folder, split, cls)
#         os.makedirs(split_folder, exist_ok=True)
    
#     # Kopyalama fonksiyonu
#     def copy_images(img_list, split):
#         for img_name in img_list:
#             src_path = os.path.join(class_folder, img_name)
#             dst_path = os.path.join(output_folder, split, cls, img_name)
#             shutil.copy2(src_path, dst_path)
    
#     # Kopyaları yap
#     copy_images(train_images, 'train')
#     copy_images(test_images, 'test')

# print("Veri başarıyla train/val/test olarak ayrıldı!")  bu kod toplam 1.5 gb lık datayı bir arabanın alttan mı aynı seviyeden mi üstten mi çekildiğine göre 3 classlı bir classification yapmak için yeterli mi