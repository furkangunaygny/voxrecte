# Cümle düzeyi puan tahmini — temel model

Özellikler: `facebook/wav2vec2-lv-60-espeak-cv-ft` (değiştirilmeden), katman başına zaman ortalaması · Tahminci: StandardScaler + Ridge · Çalıştırma: 2026-11-01T00:34:22+00:00

Veri: eğitim 2200 + doğrulama 300 kayıt (15 konuşmacı, çocuk/yetişkin oranı korunarak), test 2500 kayıt. Katman ve alpha yalnızca doğrulama kümesinde seçildi; seçilen ayar tüm eğitim verisiyle yeniden eğitilip test kümesinde bir kez ölçüldü.

| Puan | Katman | Alpha | Doğrulama PCC | Doğrulama MSE | Test PCC | Test MSE | Test MSE (ortalama tahmini) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| accuracy | 11 | 10000 | 0.702 | 0.819 | 0.678 | 1.281 | 2.372 |
| fluency | 16 | 10000 | 0.735 | 0.599 | 0.760 | 0.858 | 2.043 |
| prosodic | 16 | 10000 | 0.733 | 0.624 | 0.755 | 0.888 | 2.060 |
| total | 12 | 10000 | 0.743 | 0.677 | 0.696 | 1.238 | 2.407 |

PCC: tahmin ile uzman puanı arasındaki Pearson korelasyonu (1 = tam uyum). MSE: ortalama kare hata (0–10 ölçeğinde). Son sütun, her kayda eğitim ortalamasını veren modelin hatasıdır.

Completeness modellenmedi: 5000 kaydın 4975'inde puan 10.

## Katmana göre doğrulama PCC (en iyi alpha)

| Katman | accuracy | fluency | prosodic | total |
| --- | --- | --- | --- | --- |
| 0 | 0.613 | 0.585 | 0.544 | 0.601 |
| 1 | 0.644 | 0.568 | 0.585 | 0.633 |
| 2 | 0.648 | 0.612 | 0.629 | 0.643 |
| 3 | 0.642 | 0.624 | 0.647 | 0.644 |
| 4 | 0.651 | 0.626 | 0.654 | 0.654 |
| 5 | 0.653 | 0.662 | 0.691 | 0.659 |
| 6 | 0.671 | 0.683 | 0.706 | 0.688 |
| 7 | 0.661 | 0.693 | 0.692 | 0.690 |
| 8 | 0.664 | 0.702 | 0.697 | 0.700 |
| 9 | 0.661 | 0.702 | 0.705 | 0.698 |
| 10 | 0.691 | 0.715 | 0.729 | 0.727 |
| 11 | 0.702 | 0.733 | 0.728 | 0.742 |
| 12 | 0.702 | 0.730 | 0.727 | 0.743 |
| 13 | 0.694 | 0.733 | 0.725 | 0.736 |
| 14 | 0.694 | 0.734 | 0.727 | 0.737 |
| 15 | 0.694 | 0.731 | 0.726 | 0.734 |
| 16 | 0.694 | 0.735 | 0.733 | 0.741 |
| 17 | 0.682 | 0.732 | 0.728 | 0.729 |
| 18 | 0.667 | 0.726 | 0.728 | 0.715 |
| 19 | 0.670 | 0.730 | 0.722 | 0.712 |
| 20 | 0.653 | 0.706 | 0.699 | 0.700 |
| 21 | 0.655 | 0.694 | 0.696 | 0.698 |
| 22 | 0.640 | 0.704 | 0.697 | 0.682 |
| 23 | 0.630 | 0.671 | 0.662 | 0.664 |
| 24 | 0.635 | 0.675 | 0.669 | 0.672 |
