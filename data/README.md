# RealWaste Dataset – Açıklama Dokümanı

## 1. Veri Setinin Tanımı

Bu projede, **RealWaste** adlı açık erişimli görüntü veri seti kullanılmıştır.
RealWaste veri seti, farklı atık türlerine ait gerçek dünya görüntülerinden oluşmakta olup, atık sınıflandırma problemleri için yaygın olarak tercih edilmektedir.

Veri seti; günlük hayatta karşılaşılan geri dönüştürülebilir ve geri dönüştürülemez atık türlerini içermesi nedeniyle, çevre mühendisliği ve akıllı atık yönetimi uygulamaları açısından önemlidir.

---

## 2. Veri Setinin Kaynağı

RealWaste veri seti açık erişimlidir ve akademik çalışmalarda kullanılmasına izin verilmektedir.

Veri seti bağlantısı:
[https://archive.ics.uci.edu/dataset/908/realwaste](https://archive.ics.uci.edu/dataset/908/realwaste)

Bu proje kapsamında veri seti **sadece eğitim ve değerlendirme amaçlı** kullanılmıştır.

---

## 3. Sınıflar (Atık Türleri)

Veri seti toplam **9 farklı atık sınıfından** oluşmaktadır:

1. Cardboard (Karton)
2. Food Organics (Organik Atık)
3. Glass (Cam)
4. Metal
5. Miscellaneous Trash (Diğer Atıklar)
6. Paper (Kağıt)
7. Plastic (Plastik)
8. Textile Trash (Tekstil Atığı)
9. Vegetation (Bitkisel Atık)

Bu sınıflar, geri dönüşüm süreçlerinde sıkça karşılaşılan temel atık kategorilerini temsil etmektedir.

---

## 4. Veri Seti Yapısı

Veri seti, sınıf bazlı klasör yapısına sahiptir.
Her klasör, ilgili atık türüne ait görüntüleri içermektedir.

Proje kapsamında veri seti aşağıdaki şekilde düzenlenmiştir:

```
realwaste_split/
 ├── train/
 ├── val/
 └── test/
```

* **Train (%70)**: Modelin eğitimi için kullanılmıştır.
* **Validation (%15)**: Eğitim sırasında model performansını izlemek için kullanılmıştır.
* **Test (%15)**: Nihai performans değerlendirmesi için kullanılmıştır.

Bu bölme işlemi, modelin genelleme yeteneğini ölçmek amacıyla yapılmıştır.

---

## 5. Ön İşleme (Preprocessing)

Görüntüler, derin öğrenme modeline uygun hale getirilmek amacıyla aşağıdaki ön işlemlerden geçirilmiştir:

* Görüntülerin yeniden boyutlandırılması (224 × 224)
* Tensor dönüşümü
* ImageNet veri setine ait ortalama ve standart sapma değerleri ile normalizasyon

Bu işlemler, transfer learning yaklaşımıyla kullanılan ResNet18 modeli ile uyumluluk sağlamak amacıyla uygulanmıştır.

---

## 6. Veri Setinin Seçilme Gerekçesi

RealWaste veri setinin tercih edilme nedenleri:

* Gerçek dünya atık görüntülerini içermesi
* Açık erişimli ve akademik kullanıma uygun olması
* Çok sınıflı bir problem sunması
* Derin öğrenme tabanlı görüntü sınıflandırma çalışmalarına uygun olması

Bu özellikler, veri setini ders kapsamında gerçekleştirilen bu proje için uygun ve anlamlı kılmaktadır.

---

## 7. Not

Veri setinin boyutunun büyük olması nedeniyle, **ham görüntü dosyaları GitHub reposuna eklenmemiştir**.
Projeyi çalıştırmak isteyen kullanıcılar, veri setini yukarıda verilen bağlantı üzerinden indirerek ilgili klasör yapısına yerleştirebilirler.



