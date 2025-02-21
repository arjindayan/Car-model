# Araç Görüntü Açısı Sınıflandırma Modeli

Bu proje, araç fotoğraflarının hangi açıdan (üstten, aynı seviyeden veya alttan) çekildiğini sınıflandıran bir derin öğrenme modelini içerir. Hazır model kullanılmamıştır

## Özellikler

- Custom CNN mimarisi
- Data augmentation teknikleri
- Early stopping
- Learning rate scheduling
- L2 regularization
- Gradient clipping
- Batch normalization
- Dropout katmanları

## Gereksinimler

```bash
torch
torchvision
matplotlib
tqdm
```

## Veri Seti Yapısı

```
dataset/
├── train/
│   ├── above/
│   ├── same_level/
│   └── below/
└── test/
    ├── above/
    ├── same_level/
    └── below/
```

## Model Mimarisi

- 3 Konvolüsyonel blok
- Her blokta:
  - Conv2D katmanları
  - Batch Normalization
  - ReLU aktivasyon
  - MaxPooling
  - Dropout
- Son katmanda fully connected classifier

## Kullanım

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install torch torchvision matplotlib tqdm
```

2. Veri setini uygun formatta hazırlayın
3. Modeli eğitin:
```bash
python file.py
```

## Eğitim Parametreleri

- Batch Size: 32
- Learning Rate: 0.001
- Epochs: 70
- Early Stopping Patience: 7
- L2 Regularization: 0.001
- Weight Decay: 0.0001

## Sonuçlar

Model eğitimi sonunda:
- Loss ve accuracy grafikleri otomatik olarak gösterilir
- En iyi model 'best_model.pth' olarak kaydedilir

-MODELİN BAŞARI ORANI ANLIK %64
## Lisans

MIT

## Model Dosyası

Model dosyası (.pth) boyut sınırlaması nedeniyle GitHub'da bulunmamaktadır. Şu kaynaklardan edinebilirsiniz:
- Google Drive: https://drive.google.com/file/d/1LxJm_0cA3GTSMP5oZv8gCKT1NOZ6AVAK/view?usp=drive_link
- veya kendi modelinizi eğitebilirsiniz
