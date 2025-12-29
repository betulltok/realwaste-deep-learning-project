# Atık Görüntülerinde Derin Öğrenme ile Sınıflandırma

Bu proje, derin öğrenme yöntemleri kullanılarak atık görüntülerinin otomatik olarak sınıflandırılmasını amaçlamaktadır.
Çalışmada UCI Machine Learning Repository üzerinde yer alan **RealWaste** veri seti kullanılmış ve **ResNet18 tabanlı bir CNN modeli** ile sınıflandırma gerçekleştirilmiştir.

---

## Projenin Amacı

Günümüzde atık yönetimi ve geri dönüşüm süreçleri çevresel sürdürülebilirlik açısından büyük önem taşımaktadır.
Bu proje ile:

* Atık türlerinin otomatik olarak sınıflandırılması
* Geri dönüşüm süreçlerinin hızlandırılması
* Yapay zekâ tabanlı çevresel çözümlerin uygulanması

amaçlanmıştır.

---

## Proje Yapısı

```
realwaste-deep-learning-project/
│
├── data/               → Eğitim, doğrulama ve test verileri
├── demo_images/        → Örnek test görselleri
├── models/             → Eğitilmiş model dosyaları
├── notebooks/          → Deneysel çalışmalar
├── results/            → Model çıktıları
├── src/                → Eğitim ve yardımcı kodlar
├── gradio_app.py       → Gradio arayüzü
├── README.md
├── requirements.txt
├── .gitignore
└── .gitattributes
```

---

## Kullanılan Veri Seti

* Veri seti: RealWaste
* Kaynak: UCI Machine Learning Repository
* Veri türü: RGB görüntüler
* Sınıf sayısı: 9

### Sınıflar

* Cardboard
* Food Organics
* Glass
* Metal
* Miscellaneous Trash
* Paper
* Plastic
* Textile Trash
* Vegetation

---

## Kullanılan Yöntem

Bu projede derin öğrenme tabanlı **Convolutional Neural Network (CNN)** mimarisi kullanılmıştır.

### Model Özellikleri

* Model: ResNet18 (Transfer Learning)
* Giriş boyutu: 224 × 224
* Aktivasyon fonksiyonu: ReLU
* Kayıp fonksiyonu: Cross Entropy Loss
* Optimizasyon algoritması: Adam
* Çıkış katmanı: Softmax (9 sınıf)

---

## Model Eğitimi

* Görseller 224×224 boyutuna ölçeklendirilmiştir
* Veri seti eğitim, doğrulama ve test olarak ayrılmıştır
* Önceden eğitilmiş ResNet18 ağı kullanılmıştır
* Modelin genelleme başarısı değerlendirilmiştir

---

## Gradio Arayüzü

Proje, kullanıcıların görsel yükleyerek sınıflandırma yapabilmesi için Gradio arayüzü ile desteklenmiştir.

Uygulamayı çalıştırmak için:

```bash
python gradio_app.py
```

---

## Kurulum

```bash
pip install -r requirements.txt
```

---

## Sonuç

Bu çalışmada, görüntü tabanlı atık sınıflandırma problemi başarıyla ele alınmıştır.
Elde edilen sonuçlar, derin öğrenme yöntemlerinin çevresel uygulamalarda etkili bir şekilde kullanılabileceğini göstermektedir.
Proje, gerçek hayatta geri dönüşüm sistemlerine entegre edilebilecek bir yapı sunmaktadır.

---

## Kaynakça

[1] S. Single, S. Iranmanesh, and R. Raad, “RealWaste,” UCI Machine Learning Repository, 2023.
[2] K. He et al., “Deep Residual Learning for Image Recognition,” CVPR, 2016.
[3] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*, MIT Press, 2016.
[4] PyTorch Documentation, [https://pytorch.org](https://pytorch.org)
[5] Gradio Documentation, [https://www.gradio.app](https://www.gradio.app)

---

## Hazırlayan

Betül Tok
İstanbul Medeniyet Üniversitesi
Bilgisayar Mühendisliği Bölümü

---


