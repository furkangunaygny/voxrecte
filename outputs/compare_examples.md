# Adım 5 — Telaffuz karşılaştırma örnekleri

`scripts/compare_pronunciation.py` aracının speechocean762 train kayıtlarındaki çıktısı. Her örnekte önce uzmanların işaretlediği fonemler (fonem puanı 2'nin altında: 0 yanlış/eksik, 1 aksanlı), ardından aracın çıktısı verilir. Araç uzman puanlarını kullanmaz; yalnızca ses dosyası ve hedef metinle çalışır.

Bu örnekler yalnızca gösterim içindir; aracın başarısı Adım 6'da test kayıtlarında ölçülecek.

## 000360013 — "IT'S JUST SO HARD TO PICTURE"

Konuşmacı 0036 (f, 21 yaş) · uzman puanları: doğruluk 9, total 9 (0–10)

**Uzmanların işaretlediği fonemler:** yok (tüm fonemler doğru)

```text
Hedef cümle : IT'S JUST SO HARD TO PICTURE
Model (IPA) : ɪ z dʒ ʌ s t s oʊ h ɑːɹ d t ʊ p ɪ k tʃ ɚ
Özet        : 20 hedef fonem, 3 olası hata, 0 kabul edilebilir fark, 2/6 kelimede olası hata

Kelime         Durum                  Hedef                  Duyulan
IT'S           olası hata             IH T S                 IH Z
                 olası hata (silme): T duyulmadı
                 olası hata (değiştirme): S yerine Z (model: z)
JUST           doğru                  JH AH S T              JH AH S T
SO             doğru                  S OW                   S OW
HARD           doğru                  HH AA R D              HH AA R D
TO             olası hata             T UW                   T UH
                 olası hata (değiştirme): UW yerine UH (model: ʊ)
PICTURE        doğru                  P IH K CH ER           P IH K CH ER

Not: "olası hata" kesin bir hata kararı değildir; modelin kendi hataları da bu listede yer alabilir.
```

## 001350219 — "NO ONE WANTS TO TALK TO ME TONIGHT"

Konuşmacı 0135 (f, 20 yaş) · uzman puanları: doğruluk 10, total 9 (0–10)

**Uzmanların işaretlediği fonemler:** yok (tüm fonemler doğru)

```text
Hedef cümle : NO ONE WANTS TO TALK TO ME TONIGHT
Model (IPA) : n oʊ w ʌ n w ʌ n s t ə t ɑː k t ə m iː t ə n aɪ t
Özet        : 24 hedef fonem, 2 olası hata, 2 kabul edilebilir fark, 2/8 kelimede olası hata

Kelime         Durum                  Hedef                  Duyulan
NO             doğru                  N OW                   N OW
ONE            doğru                  W AH N                 W AH N
WANTS          olası hata             W AH N T S             W AH N S
                 olası hata (silme): T duyulmadı
TO             kabul edilebilir fark  T UW                   T AH
                 kabul edilebilir (R8): UW yerine AH (model: ə)
TALK           olası hata             T AO K                 T AA K
                 olası hata (değiştirme): AO yerine AA (model: ɑː)
TO             kabul edilebilir fark  T UW                   T AH
                 kabul edilebilir (R8): UW yerine AH (model: ə)
ME             doğru                  M IY                   M IY
TONIGHT        doğru                  T AH N AY T            T AH N AY T

Kurallar:
  R8: Referans varyantı: aynı kelime veri setinde başka bir referans yazımıyla da geçiyorsa, o yazımdaki ses kabul edilir (ör. NOT: N AH T ve N AA T).

Not: "olası hata" kesin bir hata kararı değildir; modelin kendi hataları da bu listede yer alabilir.
```

## 005940234 — "SURE WE WILL GO WITH THAT"

Konuşmacı 0594 (f, 21 yaş) · uzman puanları: doğruluk 6, total 5 (0–10)

**Uzmanların işaretlediği fonemler:** SURE: SH (0) UH (0.4) AH0 (0.4); WE: W (1.2) IY0 (1.2); THAT: AE0 (1.6)

```text
Hedef cümle : SURE WE WILL GO WITH THAT
Model (IPA) : s i v e k oː b e z t a t
Özet        : 16 hedef fonem, 13 olası hata, 0 kabul edilebilir fark, 6/6 kelimede olası hata

Kelime         Durum                  Hedef                  Duyulan
SURE           olası hata             SH UH AH               —
                 olası hata (silme): SH duyulmadı
                 olası hata (silme): UH duyulmadı
                 olası hata (silme): AH duyulmadı
WE             olası hata             W IY                   S IY
                 olası hata (değiştirme): W yerine S (model: s)
WILL           olası hata             W IH L                 V EH
                 olası hata (değiştirme): W yerine V (model: v)
                 olası hata (değiştirme): IH yerine EH (model: e)
                 olası hata (silme): L duyulmadı
GO             olası hata             G OW                   K AO
                 olası hata (değiştirme): G yerine K (model: k)
                 olası hata (değiştirme): OW yerine AO (model: oː)
WITH           olası hata             W IH DH                B EH Z
                 olası hata (değiştirme): W yerine B (model: b)
                 olası hata (değiştirme): IH yerine EH (model: e)
                 olası hata (değiştirme): DH yerine Z (model: z)
THAT           olası hata             DH AE T                T AE T
                 olası hata (değiştirme): DH yerine T (model: t)

Not: "olası hata" kesin bir hata kararı değildir; modelin kendi hataları da bu listede yer alabilir.
```

## 007390062 — "IT WAS A TOUGH POSITION TO BE IN"

Konuşmacı 0739 (m, 20 yaş) · uzman puanları: doğruluk 7, total 6 (0–10)

**Uzmanların işaretlediği fonemler:** TOUGH: T (1.6) AH0 (0.4) F (1.6); POSITION: P (1.6) AH0 (0.4) Z (0.4) IH1 (1.2) SH (1.2) N (1.2); TO: UW0 (1.8); BE: IY0 (1.8)

```text
Hedef cümle : IT WAS A TOUGH POSITION TO BE IN
Model (IPA) : ɪ ɾ w ʌ z ɐ t aʊ p aʊ l s eɪ ʃ ɪ ŋ t uː b iː ŋ
Özet        : 21 hedef fonem, 10 olası hata, 1 kabul edilebilir fark, 3/8 kelimede olası hata

Kelime         Durum                  Hedef                  Duyulan
IT             kabul edilebilir fark  IH T                   IH DX
                 kabul edilebilir (R3): T yerine DX (model: ɾ)
WAS            doğru                  W AH Z                 W AH Z
A              doğru                  AH                     AH
TOUGH          olası hata             T AH F                 T AW
                 olası hata (değiştirme): AH yerine AW (model: aʊ)
                 olası hata (silme): F duyulmadı
POSITION       olası hata             P AH Z IH SH N         P AW L S EY SH IH NG
                 olası hata (değiştirme): AH yerine AW (model: aʊ)
                 olası hata (ekleme): fazladan L (model: l)
                 olası hata (değiştirme): Z yerine S (model: s)
                 olası hata (değiştirme): IH yerine EY (model: eɪ)
                 olası hata (ekleme): fazladan IH (model: ɪ)
                 olası hata (değiştirme): N yerine NG (model: ŋ)
TO             doğru                  T UW                   T UW
BE             doğru                  B IY                   B IY
IN             olası hata             IH N                   NG
                 olası hata (silme): IH duyulmadı
                 olası hata (değiştirme): N yerine NG (model: ŋ)

Kurallar:
  R3: Dil ucu çırpması: DX, referanstaki T veya D yerine kabul edilir.

Not: "olası hata" kesin bir hata kararı değildir; modelin kendi hataları da bu listede yer alabilir.
```

## 026210157 — "HI ANGEL YOU NEVER KNOW WHAT WILL HAPPEN"

Konuşmacı 2621 (m, 19 yaş) · uzman puanları: doğruluk 7, total 7 (0–10)

**Uzmanların işaretlediği fonemler:** HI: AY0 (0.4); ANGEL: EY1 (1.6) L (1.6); WHAT: AH0 (1.8); WILL: L (1.6); HAPPEN: P (0.4)

```text
Hedef cümle : HI ANGEL YOU NEVER KNOW WHAT WILL HAPPEN
Model (IPA) : h iː æ n dʒ ɚ j uː n ɛ v ɚ n oʊ w ʌ t w ɪ ɑːɹ h æ m b ə n
Özet        : 25 hedef fonem, 8 olası hata, 0 kabul edilebilir fark, 4/8 kelimede olası hata

Kelime         Durum                  Hedef                  Duyulan
HI             olası hata             HH AY                  HH IY
                 olası hata (değiştirme): AY yerine IY (model: iː)
ANGEL          olası hata             EY N JH L              AE N JH ER
                 olası hata (değiştirme): EY yerine AE (model: æ)
                 olası hata (ekleme): fazladan ER (model: ɚ)
                 olası hata (silme): L duyulmadı
YOU            doğru                  Y UW                   Y UW
NEVER          doğru                  N EH V ER              N EH V ER
KNOW           doğru                  N OW                   N OW
WHAT           doğru                  W AH T                 W AH T
WILL           olası hata             W IH L                 W IH AA R
                 olası hata (ekleme): fazladan AA (model: ɑːɹ)
                 olası hata (değiştirme): L yerine R (model: ɑːɹ)
HAPPEN         olası hata             HH AE P AH N           HH AE M B AH N
                 olası hata (ekleme): fazladan M (model: m)
                 olası hata (değiştirme): P yerine B (model: b)

Not: "olası hata" kesin bir hata kararı değildir; modelin kendi hataları da bu listede yer alabilir.
```
