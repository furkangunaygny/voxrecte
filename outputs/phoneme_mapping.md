# Adım 4 — Referans ve model fonemlerinin ortak biçimi

Teslim: dönüşüm kodu (`scripts/phoneme_map.py`), sembol eşleştirme tablosu (bu rapor ve `phoneme_mapping.csv`) ve test örnekleri (`tests/test_phoneme_map.py` ve aşağıdaki kelime örnekleri). Bu rapor `scripts/check_phoneme_map.py` tarafından yalnızca **train** kayıtları kullanılarak üretilir; test kayıtları Adım 6'daki değerlendirmeye ayrıldı.

## Özet

- Ortak gösterim: CMU sözlüğünün **39 ARPAbet fonemi** (vurgusuz) + modelin ürettiği ama ARPAbet'te karşılığı olmayan **3 ses** (DX, Q, IX).
- Model train kayıtlarında 136 farklı sembol üretti; sembollerin %94.63'i doğrudan tablodan, %4.78'i indirgenerek eşlendi, %0.59'i İngilizce karşılığı olmadığı için `?` olarak işaretlendi.
- Uzmanların tüm fonemleri doğru saydığı **222 temiz kayıtta** fonem hata oranı kurallar olmadan **%21.7**, kabul edilebilir fark kurallarıyla **%17.4**. Kalan fark, modelin kendi hataları ile henüz kurala bağlanmamış farklardan oluşur (aşağıda).
- Tüm train kayıtlarında (2500): kurallar olmadan %36.7, kurallarla %32.3. Bu fark gerçek telaffuz hatalarını da içerir.

## 1. Ortak gösterim ve vurgu

- Referans (speechocean762): ARPAbet, ünlülerde vurgu rakamı (0 vurgusuz, 1 birincil, 2 ikincil). Rakam fonemden ayrılır ve ayrı tutulur.
- Model: eSpeak IPA; vurgu işareti üretmez, kelime sınırı vermez. Her sembol tablodaki bir veya birkaç ortak foneme çevrilir (ör. `ɑːɹ` → `AA R`).
- Vurgu fonem eşleşmesinde karşılaştırılmaz (R1); yalnızca vurgusuz ünlülerin indirgenmesine izin vermek için kullanılır (R2). Kelime vurgusu speechocean762'de ayrıca puanlandığı için fonem düzeyinin dışında kalır.

## 2. Kabul edilebilir telaffuz farkları

Bu farklar hata sayılmaz; karşılaştırma adımında ayrıca gösterilir.

| Kural | Açıklama | Temiz kayıtlarda kullanım |
| --- | --- | --- |
| R1 | Vurgu karşılaştırılmaz: model vurgu işareti üretmez, referanstaki vurgu rakamı yalnızca R2 ve R7'de kullanılır. Tek heceli kelimelerde rakam bilgi taşımadığı için (veride %98'i 0) vurgu bilinmiyor sayılır. | — |
| R2 | Vurgusuz ünlü indirgenmesi: çok heceli bir kelimede vurgusuz (0) ünlü yerine AH, IH veya IX kabul edilir. | 14 |
| R3 | Dil ucu çırpması: DX, referanstaki T veya D yerine kabul edilir. | 24 |
| R4 | Gırtlak durması: Q, referanstaki T yerine kabul edilir. | 0 |
| R5 | r'li/r'siz aksan: ünlüden sonra gelen R referansta olup tahminde olmayabilir ya da tersi. | 43 |
| R6 | r'li/r'siz aksan: çok heceli bir kelimenin vurgusuz hecesinde ER ile AH birbirinin yerine kabul edilir ("mother": M AH DH AH / m ʌ ð ɚ). | 3 |
| R7 | Hecelik ünsüz: L, N veya M'den önce vurgusuz AH olmayabilir ("little" → l ɪ ɾ əl / L IH T L). | 4 |
| R8 | Referans varyantı: aynı kelime veri setinde başka bir referans yazımıyla da geçiyorsa, o yazımdaki ses kabul edilir (ör. NOT: N AH T ve N AA T). | 60 |

R5 ve R6, speechocean762 referanslarının kayıt bazında İngiliz (r'siz) veya Amerikan (r'li) yazılmasından dolayı gerekli: ör. "FOR" 278 kayıtta `F AO`, 40 kayıtta `F AO R`; model ise Amerikan tarzında `f ɔːɹ` üretir.

## 3. Sembol eşleştirme tablosu

Son sütun, sembolün model tarafından train kayıtlarında kaç kez üretildiği.

| Model (IPA) | Ortak | Train'de | | Model (IPA) | Ortak | Train'de |
| --- | --- | --- | --- | --- | --- | --- |
| `i` | IY | 770 | | `t` | T | 3316 |
| `iː` | IY | 1097 | | `d` | D | 1594 |
| `ɪ` | IH | 2078 | | `k` | K | 1423 |
| `ɪː` | IY | — | | `g` | G | — |
| `ᵻ` | IX | 134 | | `ɡ` | G | 473 |
| `e` | EH | 127 | | `f` | F | 871 |
| `eː` | EH | 127 | | `v` | V | 731 |
| `ɛ` | EH | 987 | | `θ` | TH | 110 |
| `ɛː` | EH | — | | `ð` | DH | 963 |
| `æ` | AE | 1424 | | `s` | S | 2303 |
| `a` | AE | 507 | | `z` | Z | 1483 |
| `aː` | AA | 60 | | `ʃ` | SH | 290 |
| `ɑ` | AA | 235 | | `ʒ` | ZH | 17 |
| `ɑː` | AA | 542 | | `h` | HH | 846 |
| `ɒ` | AA | — | | `tʃ` | CH | 182 |
| `ɔ` | AO | 266 | | `dʒ` | JH | 211 |
| `ɔː` | AO | 208 | | `m` | M | 1471 |
| `o` | OW | 100 | | `n` | N | 2722 |
| `oː` | AO | 121 | | `ŋ` | NG | 700 |
| `ʊ` | UH | 204 | | `l` | L | 1285 |
| `u` | UW | 43 | | `ɫ` | L | — |
| `uː` | UW | 1122 | | `ɹ` | R | 966 |
| `ʌ` | AH | 1364 | | `r` | R | 80 |
| `ə` | AH | 1559 | | `ɻ` | R | — |
| `ɐ` | AH | 495 | | `w` | W | 1470 |
| `ɚ` | ER | 849 | | `j` | Y | 698 |
| `ɜ` | ER | 8 | | `ɾ` | DX | 278 |
| `ɜː` | ER | 276 | | `ʔ` | Q | 10 |
| `eɪ` | EY | 929 | | `ph` | P | 44 |
| `aɪ` | AY | 941 | | `th` | T | 235 |
| `aʊ` | AW | 318 | | `kh` | K | 95 |
| `ɔɪ` | OY | — | | `x` | HH | 291 |
| `oʊ` | OW | 715 | | `tɕ` | JH | 43 |
| `əʊ` | OW | — | | `tɕh` | CH | 12 |
| `ɑːɹ` | AA R | 227 | | `ts.h` | CH | 24 |
| `ɔːɹ` | AO R | 331 | | `ɕ` | SH | 39 |
| `oːɹ` | AO R | 23 | | `ʉ` | UW | 60 |
| `ɪɹ` | IH R | 10 | | `ai` | AY | — |
| `ɛɹ` | EH R | 7 | | `ei` | EY | — |
| `ʊɹ` | UH R | — | | `au` | AW | — |
| `aɪɚ` | AY ER | — | | `ɑu` | AW | — |
| `aɪə` | AY AH | — | | `ou` | OW | — |
| `iə` | IY AH | — | | `oɪ` | OY | 14 |
| `eə` | EH AH | — | | `iɛ` | Y EH | — |
| `ʊə` | UH AH | — | | `iɑ` | Y AA | — |
| `əl` | AH L | 138 | | `iou` | Y OW | — |
| `l̩` | AH L | — | | `uo` | W AO | — |
| `n̩` | AH N | — | | `ua` | W AA | — |
| `p` | P | 784 | | `uei` | W EY | — |
| `b` | B | 792 | | `onɡ` | UH NG | — |

Tabloda olmayan semboller (modelin çok dilli sözlüğünden): uzunluk işareti, ton rakamı ve ikincil artikülasyon (ʰ, ʲ) atılarak tekrar aranır; yine bulunamazsa `?` olur.

| Train'de görülen tablo dışı sembol | Eşleşme | Adet |
| --- | --- | --- |
| `i5` | IY (indirgenmiş) | 320 |
| `ɑ5` | AA (indirgenmiş) | 242 |
| `ə1` | AH (indirgenmiş) | 220 |
| `u5` | UW (indirgenmiş) | 184 |
| `ai5` | AY (indirgenmiş) | 167 |
| `ei5` | EY (indirgenmiş) | 141 |
| `ɑu5` | AW (indirgenmiş) | 134 |
| `o5` | OW (indirgenmiş) | 122 |
| `a5` | AE (indirgenmiş) | 117 |
| `ts` | ? (bilinmiyor) | 103 |
| `əɜ` | AH (indirgenmiş) | 93 |
| `ou5` | OW (indirgenmiş) | 73 |
| `i̪5` | IY (indirgenmiş) | 72 |
| `i.5` | IY (indirgenmiş) | 66 |
| `iɛ5` | Y EH (indirgenmiş) | 61 |
| `s.` | S (indirgenmiş) | 57 |
| `uo5` | W AO (indirgenmiş) | 48 |
| `y5` | ? (bilinmiyor) | 33 |
| `iɜ` | IY (indirgenmiş) | 31 |
| `ts.` | ? (bilinmiyor) | 29 |
| `oɜ` | OW (indirgenmiş) | 29 |
| `ʐ` | ? (bilinmiyor) | 27 |
| `onɡ5` | UH NG (indirgenmiş) | 21 |
| `ə5` | AH (indirgenmiş) | 17 |
| `iɑ5` | Y AA (indirgenmiş) | 15 |
| `iou2` | Y OW (indirgenmiş) | 11 |
| `ɲ` | ? (bilinmiyor) | 10 |
| `iouɜ` | Y OW (indirgenmiş) | 10 |
| `uoɜ` | W AO (indirgenmiş) | 9 |
| `ɑ̃` | AA (indirgenmiş) | 8 |
| `yː` | ? (bilinmiyor) | 7 |
| `uei5` | W EY (indirgenmiş) | 7 |
| `q` | ? (bilinmiyor) | 7 |
| `tsh` | ? (bilinmiyor) | 7 |
| `y` | ? (bilinmiyor) | 6 |
| `øː` | ? (bilinmiyor) | 6 |
| `ʲ` | ? (bilinmiyor) | 6 |
| `ɯ` | ? (bilinmiyor) | 5 |
| `ɔɨ` | ? (bilinmiyor) | 5 |
| `ʁ` | ? (bilinmiyor) | 5 |
| `uɨ` | ? (bilinmiyor) | 4 |
| `ħ` | ? (bilinmiyor) | 3 |
| `ø` | ? (bilinmiyor) | 3 |
| `œ` | ? (bilinmiyor) | 3 |
| `ɨ` | ? (bilinmiyor) | 2 |
| `ɐ̃ʊ̃` | ? (bilinmiyor) | 2 |
| `t[` | T (indirgenmiş) | 2 |
| `yɜ` | ? (bilinmiyor) | 2 |
| `ɬ` | ? (bilinmiyor) | 2 |
| `rʲ` | R (indirgenmiş) | 1 |
| `mʲ` | M (indirgenmiş) | 1 |
| `ʎ` | ? (bilinmiyor) | 1 |
| `ɖ` | ? (bilinmiyor) | 1 |
| `ɵ` | ? (bilinmiyor) | 1 |
| `ũ` | UW (indirgenmiş) | 1 |
| `ɨː` | ? (bilinmiyor) | 1 |
| `ɔ̃` | AO (indirgenmiş) | 1 |
| `ɛ̃` | EH (indirgenmiş) | 1 |
| `aiɜ` | AY (indirgenmiş) | 1 |
| `ç` | ? (bilinmiyor) | 1 |
| `o2` | OW (indirgenmiş) | 1 |
| `ə2` | AH (indirgenmiş) | 1 |
| `əɪ` | ? (bilinmiyor) | 1 |
| `ɭ` | ? (bilinmiyor) | 1 |

## 4. Kelime örnekleri

Temiz kayıtlardan, her kural için gerçek model çıktısı. Referans vurgu rakamlarıyla, model çıktısı ortak gösterimle verilmiştir.

| Kural | Kelime | Referans | Model | Kayıt |
| --- | --- | --- | --- | --- |
| R2 | REPORTS | R IH0 P AO1 T S | R IX P AO R T S | 021680237 |
| R2 | REMINDER | R IH0 M AY1 N D AH0 | R IX M AY N D ER | 022420138 |
| R2 | DEFEND | D IH0 F EH1 N D | D IX F EH N D | 023800278 |
| R2 | VERY | V EH1 R IY0 | V EH R IH | 055510165 |
| R3 | MOTOR | M OW1 T ER0 | M OW DX ER | 000060106 |
| R3 | STARTING | S T AA1 R T IH0 NG | S T AA R DX IH NG | 000360210 |
| R3 | PARTY | P AA1 R T IY0 | P AA R DX IY | 013090181 |
| R3 | NOTICE | N OW1 T IH0 S | N OW DX IH S | 020360019 |
| R5 | FOUR | F AO0 | F AO R | 000540031 |
| R5 | YOUR | Y AO0 | Y AO R | 001310076 |
| R5 | MORE | M AO0 | M AO R | 005940019 |
| R5 | FOR | F AO0 | F AO R | 007390120 |
| R6 | REMINDER | R IH0 M AY1 N D AH0 | R IX M AY N D ER | 022420138 |
| R6 | SUCCESSFULLY | S AH0 K S EH1 S F AH0 L IY0 | S AH K S EH S F ER L IY | 023800278 |
| R6 | FORGET | F AH0 G EH0 T | F ER G EH T | 054140098 |
| R7 | FAMILY | F AE1 M AH0 L IY0 | F AE M L IY | 054240053 |
| R8 | TO | T UW0 | T AH | 000060015 |
| R8 | DO | D UH0 | D UW | 000260126 |
| R8 | WANT | W AA0 N T | W AH N T | 000260126 |
| R8 | NOT | N AH0 T | N AA T | 000360210 |

## 5. Kurallardan sonra kalan en sık farklar (temiz kayıtlar)

Bu kayıtlarda uzmanlar her fonemi doğru saydığı için kalan farklar çoğunlukla modelin hatası ya da kurala bağlanmamış bir varyanttır. Yeni kural eklemeden önce incelenmeli.

| Değiştirme (referans → model) | Adet | Silme (referansta var, modelde yok) | Adet | Ekleme (modelde fazla) | Adet |
| --- | --- | --- | --- | --- | --- |
| IH → IY | 23 | T | 12 | AH | 43 |
| AH → AA | 18 | IH | 9 | ER | 12 |
| AH → AO | 16 | N | 8 | Y | 8 |
| UH → UW | 16 | L | 7 | N | 7 |
| N → NG | 14 | Y | 6 | S | 5 |
| AH → AE | 14 | AH | 6 | EH | 5 |
| DH → Z | 12 | NG | 3 | Z | 5 |
| IH → EH | 10 | AE | 3 | T | 4 |
| IY → IH | 9 | IY | 3 | IY | 4 |
| AE → EH | 8 | S | 2 | IX | 4 |

## Sınırlılıklar

- R5 ve R6 bağlamdan bağımsızdır: r'siz okunuş her yerde kabul edilir. Bu, gerçek bir R hatasını da gizleyebilir; Türk konuşmacı verisinde hedef aksan belirlenince daraltılabilir.
- Fonksiyon kelimelerinin zayıf biçimleri (ör. AND → `AH N`) ayrıca kurala bağlanmadı; R2 ünlü kısmını karşılar, ünsüz düşmesini karşılamaz.
- Kontrol, modelin kendi hatalarını kural eksiklerinden ayıramaz; kalan farklar elle incelenmelidir.
- Referansın İngiliz aksanındaki "o" sesini (ɒ) `AH` ile yazdığı kelimeler, veri setinde başka bir yazımı yoksa (ör. JOHN, POCKET) R8 kapsamına girmez; model bunlarda Amerikan `AA`/`AO` üretir. Hedef aksan belirlenince bu kelimeler için kural genişletilebilir.
- Kalan farkların bir kısmı uzmanların hoş gördüğü aksan özellikleri olabilir (ör. GOOD/LOOK'ta `UH → UW`, kelime sonuna eklenen `AH`). Model bunları duyuyor ama uzman "doğru" saymış; Adım 6'da değerlendirme ölçütü seçilirken dikkate alınmalı.
