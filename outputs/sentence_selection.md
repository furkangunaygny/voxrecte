# Adım 3 — Okutulacak cümlelerin seçimi

Teslim: cümle listesi, kaynak kayıt kimlikleri ve fonem kapsama tablosu. Bu rapor `scripts/select_sentences.py` tarafından üretilir; aynı veriyle her çalıştırmada aynı sonucu verir.

## Özet

- speechocean762'deki 5000 kayıttan **4947 tekrarsız cümle** çıkarıldı.
- **20 temel + 5 yedek cümle** algoritmik olarak seçildi. Temel cümleler 39 İngilizce fonemin **39**'unu kapsıyor.
- Her fonem en az **3 temel cümlede** geçiyor (yedeklerle en az 4). Bu nedenle tek bir temel cümle okunamazsa kapsam düşmez. En az geçen fonem: OY.
- Pilot kayıtlar (Adım 8: 5 kişi × 5 cümle) için seçilen **5 temel cümle** 39 fonemin 39'unu kapsıyor. Temel set toplam 182 kelime ve 629 fonemden oluşuyor.

## 1. Cümle listesi

| # | Rol | Pilot sırası | Cümle | Kelime | Fonem | Farklı fonem |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | temel | 1 | AMERICA'S GOING THROUGH JUST ABOUT THE SAME EXPERIENCE AS MYSELF | 10 | 46 | 25 |
| 2 | temel | 2 | DO YOU HAVE ANY QUESTIONS FOR HER | 7 | 22 | 19 |
| 3 | temel | 3 | HOWEVER MARKET CONDITIONS HAVE LEFT US WITH NO OTHER CHOICE | 10 | 38 | 24 |
| 4 | temel | 4 | THERE'S NOTHING THAT UNUSUAL ABOUT THE STRUCTURE OF THE PLAN | 10 | 40 | 22 |
| 5 | temel |  | THEY WERE PROBABLY DATING OR SOMETHING HE SHRUGGED | 8 | 30 | 21 |
| 6 | temel |  | HE SHOULD HAVE LET THE ORGANIZATION AND THE MANAGER KNOW | 10 | 35 | 22 |
| 7 | temel |  | I JUST WANT TO SEE HER SAFE IN YOUR HANDS | 10 | 30 | 20 |
| 8 | temel |  | THIS IS NOT AN ARGUMENT FOR BUSINESS AS USUAL | 9 | 33 | 18 |
| 9 | temel |  | SO ENJOY AND TELL ME WHAT YOU THINK OF IT | 10 | 27 | 21 |
| 10 | temel |  | BUT THAT WOULD NOT ALONE EXPLAIN HIS PLEASURE | 8 | 31 | 21 |
| 11 | temel |  | YOUR SECOND SHOT WILL BE A CHOICE OF TWO OPTIONS | 10 | 29 | 21 |
| 12 | temel |  | THE MOST POWERFUL THING YOU CAN DO IS RECOVER | 9 | 29 | 24 |
| 13 | temel | 5 | YET WHY THERE SHOULD BE NOTHING IS HARD TO SAY | 10 | 28 | 23 |
| 14 | temel |  | YOU HAVE TO CHANGE WITH THEM OR BE LEFT BEHIND | 10 | 30 | 23 |
| 15 | temel |  | I SHALL NEVER FORGIVE MYSELF FOR THIS EXPOSURE | 8 | 31 | 19 |
| 16 | temel |  | YOU HAVE DONE THAT WHICH MUST MAKE A FATHER ANGRY | 10 | 31 | 23 |
| 17 | temel |  | THE SOLDIER STARED INTO SPACE MIND RACING FURIOUSLY | 8 | 38 | 23 |
| 18 | temel |  | SHE OPENED THE DOOR AND LOOKED AROUND THE BATHROOM | 9 | 30 | 21 |
| 19 | temel |  | BOTH GIRLS CALLED OUT AS THEY WATCHED HER LEAVE | 9 | 26 | 21 |
| 20 | temel |  | HER EYES USUALLY SO GENTLE FLASHED FIRE | 7 | 25 | 19 |
| 21 | yedek |  | THANKS FOR THE JOY OF OWNERSHIP | 6 | 19 | 17 |
| 22 | yedek |  | COME IT WOULD GIVE ME SO MUCH PLEASURE | 8 | 23 | 19 |
| 23 | yedek |  | I AM RICH AND SO WAS MY FATHER BEFORE ME | 10 | 27 | 22 |
| 24 | yedek |  | I THINK THE SHAKING HURT HIS FEELINGS VERY MUCH | 9 | 31 | 22 |
| 25 | yedek |  | ANGER IS SOMETHING THAT CAN BE REALLY HARD TO CONTROL | 10 | 38 | 23 |

<details><summary>Cümlelerin referans fonemleri (ARPAbet, vurgu rakamları atılmış; · kelime sınırı)</summary>

| # | Cümle | Fonemler |
| --- | --- | --- |
| 1 | AMERICA'S GOING THROUGH JUST ABOUT THE SAME EXPERIENCE AS MYSELF | AH M EH R IH K AH S · G OW IH NG · TH R UW · JH AH S T · AH B AW T · DH AH · S EY M · IH K S P IH AH IH AH N S · AE Z · M AY S EH L F |
| 2 | DO YOU HAVE ANY QUESTIONS FOR HER | D UH · Y UW · HH AE V · EH N IY · K W EH S CH AH N Z · F AO · HH ER |
| 3 | HOWEVER MARKET CONDITIONS HAVE LEFT US WITH NO OTHER CHOICE | HH AW EH V ER · M AA K IH T · K AH N D IH SH N Z · HH AE V · L EH F T · AH S · W IH DH · N OW · AH DH ER · CH OY S |
| 4 | THERE'S NOTHING THAT UNUSUAL ABOUT THE STRUCTURE OF THE PLAN | DH EH R Z · N AH TH IH NG · DH AE T · AH N Y UW ZH AH L · AH B AW T · DH AH · S T R AH K CH AH · AH V · DH AH · P L AE N |
| 5 | THEY WERE PROBABLY DATING OR SOMETHING HE SHRUGGED | DH EY · W ER · P R AH B AH B L IY · D EY T IH NG · AO · S AH M TH IH NG · HH IY · SH R AH G |
| 6 | HE SHOULD HAVE LET THE ORGANIZATION AND THE MANAGER KNOW | HH IY · SH UH D · HH AE V · L EH T · DH AH · AO G AH N AY Z EY SH N · AE N D · DH AH · M AE N IH JH AH · N OW |
| 7 | I JUST WANT TO SEE HER SAFE IN YOUR HANDS | AY · JH AH S T · W AA N T · T UW · S IY · HH ER · S EY F · IH N · Y UH AH · HH AE N D Z AA N |
| 8 | THIS IS NOT AN ARGUMENT FOR BUSINESS AS USUAL | DH IH S · IH Z · N AH T · AE N · AA G Y UW M AH N T · F AO · B IH Z N AH S · AE Z · Y UW ZH AH L |
| 9 | SO ENJOY AND TELL ME WHAT YOU THINK OF IT | S OW · IH N JH OY · AE N D · T EH L · M IY · W AH T · Y UW · TH IH NG K · AH V · IH T |
| 10 | BUT THAT WOULD NOT ALONE EXPLAIN HIS PLEASURE | B AH T · DH AE T · W UH D · N AH T · AH L OW N · IH K S P L EY N · HH IH Z · P L EH ZH ER |
| 11 | YOUR SECOND SHOT WILL BE A CHOICE OF TWO OPTIONS | Y AO · S EH K AH N D · SH AH T · W IH L · B IY · AH · CH OY S · AH V · T UW · AH P SH N Z |
| 12 | THE MOST POWERFUL THING YOU CAN DO IS RECOVER | DH AH · M OW S T · P AW AH F L · TH IH NG · Y UW · K AE N · D UH · IH Z · R IH K AH V ER |
| 13 | YET WHY THERE SHOULD BE NOTHING IS HARD TO SAY | Y EH T · W AY · DH EH R · SH UH D · B IY · N AH TH IH NG · IH Z · HH AA R D · T UW · S EY |
| 14 | YOU HAVE TO CHANGE WITH THEM OR BE LEFT BEHIND | Y UW · HH AE V · T UW · CH EY N JH · W IH TH · DH EH M · AO · B IY · L EH F T · B IH HH AY N D |
| 15 | I SHALL NEVER FORGIVE MYSELF FOR THIS EXPOSURE | AY · SH AH L · N EH V ER · F AH G IH V · M AY S EH L F · F AO · DH IH S · IH K S P OW ZH AH |
| 16 | YOU HAVE DONE THAT WHICH MUST MAKE A FATHER ANGRY | Y UW · HH AE V · D AH N · DH AE T · W IH CH · M AH S T · M EY K · AH · F AA DH AH · AE NG G R IY |
| 17 | THE SOLDIER STARED INTO SPACE MIND RACING FURIOUSLY | DH AH · S OW L JH ER · S T EH ER D · IH N T UW · S P EY S · M AY N D · R EY S IH NG · F Y UH ER IH AH S L IY |
| 18 | SHE OPENED THE DOOR AND LOOKED AROUND THE BATHROOM | SH IY · OW P AH N · DH AH · D AO · AE N D · L UH K T · AH R AW N D · DH AH · B AA TH R UW M |
| 19 | BOTH GIRLS CALLED OUT AS THEY WATCHED HER LEAVE | B OW TH · G ER L Z · K AO L D · AW T · AE Z · DH EY · W AH CH T · HH ER · L IY V |
| 20 | HER EYES USUALLY SO GENTLE FLASHED FIRE | HH ER · AY Z · Y UW ZH AH L IY · S OW · JH EH N T L · F L AE SH T · F AY AH |
| 21 | THANKS FOR THE JOY OF OWNERSHIP | TH AE NG K S · F AO · DH AH · JH OY · AH V · OW N AH SH IH P |
| 22 | COME IT WOULD GIVE ME SO MUCH PLEASURE | K AH M · IH T · W UH D · G IH V · M IY · S OW · M AH CH · P L EH ZH ER |
| 23 | I AM RICH AND SO WAS MY FATHER BEFORE ME | AY · EY EH M · R IH CH · AE N D · S OW · W AH Z · M AY · F AA DH ER · B IH F AO · M IY |
| 24 | I THINK THE SHAKING HURT HIS FEELINGS VERY MUCH | AY · TH IH NG K · DH AH · SH EY K IH NG · HH ER T · HH IH Z · F IY L IY NG S · V EH R IY · M AH CH |
| 25 | ANGER IS SOMETHING THAT CAN BE REALLY HARD TO CONTROL | AE NG G ER · IH Z · S AH M TH IH NG · DH AE T · K AE N · B IY · R IH AH L IY · HH AA R D · T UW · K AH N T R OW L |

</details>

## 2. Kaynak kayıt kimlikleri

Her cümlenin speechocean762'de hangi kayıtta geçtiği. Kaynak kayıtlar anadili Mandarin olan konuşmacılara aittir; yalnızca cümlenin kaynağını ve referans transkripsiyonunu belirtir.

| # | Rol | Cümle | Kayıt kimliği (utt_id) | Bölüm | Konuşmacı (cinsiyet, yaş, grup) | Uzman total puanı |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | temel | AMERICA'S GOING THROUGH JUST ABOUT THE SAME EXPERIENCE AS MYSELF | 090880095 | test | 9088 (f, 15, child) | 8 |
| 2 | temel | DO YOU HAVE ANY QUESTIONS FOR HER | 012930139 | test | 1293 (m, 32, adult) | 8 |
| 3 | temel | HOWEVER MARKET CONDITIONS HAVE LEFT US WITH NO OTHER CHOICE | 007650036 | test | 0765 (m, 20, adult) | 6 |
| 4 | temel | THERE'S NOTHING THAT UNUSUAL ABOUT THE STRUCTURE OF THE PLAN | 022360321 | train | 2236 (m, 22, adult) | 7 |
| 5 | temel | THEY WERE PROBABLY DATING OR SOMETHING HE SHRUGGED | 096470012 | test | 9647 (f, 20, adult) | 3 |
| 6 | temel | HE SHOULD HAVE LET THE ORGANIZATION AND THE MANAGER KNOW | 023800114 | train | 2380 (f, 21, adult) | 8 |
| 7 | temel | I JUST WANT TO SEE HER SAFE IN YOUR HANDS | 096180018 | test | 9618 (f, 24, adult) | 6 |
| 8 | temel | THIS IS NOT AN ARGUMENT FOR BUSINESS AS USUAL | 029370106 | test | 2937 (m, 19, adult) | 8 |
| 9 | temel | SO ENJOY AND TELL ME WHAT YOU THINK OF IT | 096250021 | train | 9625 (f, 25, adult) | 3 |
| 10 | temel | BUT THAT WOULD NOT ALONE EXPLAIN HIS PLEASURE | 095530066 | test | 9553 (f, 15, child) | 8 |
| 11 | temel | YOUR SECOND SHOT WILL BE A CHOICE OF TWO OPTIONS | 004820114 | train | 0482 (m, 28, adult) | 8 |
| 12 | temel | THE MOST POWERFUL THING YOU CAN DO IS RECOVER | 023550201 | test | 2355 (m, 20, adult) | 9 |
| 13 | temel | YET WHY THERE SHOULD BE NOTHING IS HARD TO SAY | 081530100 | test | 8153 (f, 14, child) | 8 |
| 14 | temel | YOU HAVE TO CHANGE WITH THEM OR BE LEFT BEHIND | 096080027 | test | 9608 (f, 30, adult) | 4 |
| 15 | temel | I SHALL NEVER FORGIVE MYSELF FOR THIS EXPOSURE | 004820378 | train | 0482 (m, 28, adult) | 8 |
| 16 | temel | YOU HAVE DONE THAT WHICH MUST MAKE A FATHER ANGRY | 054080081 | train | 5408 (m, 11, child) | 8 |
| 17 | temel | THE SOLDIER STARED INTO SPACE MIND RACING FURIOUSLY | 096450011 | train | 9645 (f, 20, adult) | 4 |
| 18 | temel | SHE OPENED THE DOOR AND LOOKED AROUND THE BATHROOM | 096000018 | train | 9600 (f, 28, adult) | 5 |
| 19 | temel | BOTH GIRLS CALLED OUT AS THEY WATCHED HER LEAVE | 096470005 | test | 9647 (f, 20, adult) | 4 |
| 20 | temel | HER EYES USUALLY SO GENTLE FLASHED FIRE | 096400016 | test | 9640 (f, 29, adult) | 4 |
| 21 | yedek | THANKS FOR THE JOY OF OWNERSHIP | 020590121 | train | 2059 (f, 21, adult) | 8 |
| 22 | yedek | COME IT WOULD GIVE ME SO MUCH PLEASURE | 091010151 | test | 9101 (f, 15, child) | 9 |
| 23 | yedek | I AM RICH AND SO WAS MY FATHER BEFORE ME | 096040023 | train | 9604 (f, 21, adult) | 4 |
| 24 | yedek | I THINK THE SHAKING HURT HIS FEELINGS VERY MUCH | 052180112 | train | 5218 (f, 12, child) | 8 |
| 25 | yedek | ANGER IS SOMETHING THAT CAN BE REALLY HARD TO CONTROL | 023550230 | test | 2355 (m, 20, adult) | 9 |

## 3. Fonem kapsama tablosu

Satırlar fonemin tüm veri setindeki sıklığına göre sıralıdır. "Adet" ve "%" fonem sayısını, "cümle" sütunları fonemin kaç farklı cümlede geçtiğini gösterir. Son sütun, fonemin geçtiği cümlelerin 1. tablodaki numaralarıdır.

| Fonem | Tür | Veri seti (adet) | Veri seti % | Temel (adet) | Temel % | Temel cümle | Temel+yedek cümle | Geçtiği cümleler (#) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AH | ünlü | 8706 | 9.33 | 64 | 10.17 | 19 | 24 | 1 2 3 4 5 6 7 8 9 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 |
| T | ünsüz | 7891 | 8.45 | 36 | 5.72 | 18 | 21 | 1 3 4 5 6 7 8 9 10 11 12 13 14 16 17 18 19 20 22 24 25 |
| N | ünsüz | 5954 | 6.38 | 41 | 6.52 | 18 | 21 | 1 2 3 4 6 7 8 9 10 11 12 13 14 15 16 17 18 20 21 23 25 |
| IH | ünlü | 5835 | 6.25 | 36 | 5.72 | 16 | 21 | 1 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 21 22 23 24 25 |
| S | ünsüz | 4186 | 4.48 | 33 | 5.25 | 16 | 21 | 1 2 3 4 5 7 8 9 10 11 12 13 15 16 17 20 21 22 23 24 25 |
| IY | ünlü | 3672 | 3.93 | 14 | 2.23 | 13 | 17 | 2 5 6 7 9 11 13 14 16 17 18 19 20 22 23 24 25 |
| D | ünsüz | 3631 | 3.89 | 20 | 3.18 | 15 | 18 | 2 3 5 6 7 9 10 11 12 13 14 16 17 18 19 22 23 25 |
| L | ünsüz | 3595 | 3.85 | 25 | 3.97 | 16 | 19 | 1 3 4 5 6 8 9 10 11 12 14 15 17 18 19 20 22 24 25 |
| W | ünsüz | 3117 | 3.34 | 11 | 1.75 | 11 | 13 | 2 3 5 7 9 10 11 13 14 16 19 22 23 |
| DH | ünsüz | 2995 | 3.21 | 22 | 3.50 | 15 | 19 | 1 3 4 5 6 8 10 12 13 14 15 16 17 18 19 21 23 24 25 |
| M | ünsüz | 2939 | 3.15 | 15 | 2.38 | 12 | 16 | 1 3 5 6 8 9 12 14 15 16 17 18 22 23 24 25 |
| AE | ünlü | 2834 | 3.04 | 21 | 3.34 | 15 | 18 | 1 2 3 4 6 7 8 9 10 12 14 16 18 19 20 21 23 25 |
| K | ünsüz | 2780 | 2.98 | 15 | 2.38 | 12 | 16 | 1 2 3 4 9 10 11 12 15 16 18 19 21 22 24 25 |
| UW | ünlü | 2732 | 2.93 | 16 | 2.54 | 14 | 15 | 1 2 4 7 8 9 11 12 13 14 16 17 18 20 25 |
| EH | ünlü | 2506 | 2.68 | 19 | 3.02 | 13 | 16 | 1 2 3 4 6 9 10 11 13 14 15 17 20 22 23 24 |
| Z | ünsüz | 2501 | 2.68 | 16 | 2.54 | 13 | 16 | 1 2 3 4 6 7 8 10 11 12 13 19 20 23 24 25 |
| AY | ünlü | 2449 | 2.62 | 10 | 1.59 | 8 | 10 | 1 6 7 13 14 15 17 20 23 24 |
| R | ünsüz | 2405 | 2.58 | 13 | 2.07 | 8 | 11 | 1 4 5 12 13 16 17 18 23 24 25 |
| HH | ünsüz | 2160 | 2.31 | 16 | 2.54 | 11 | 13 | 2 3 5 6 7 10 13 14 16 19 20 24 25 |
| B | ünsüz | 1731 | 1.85 | 12 | 1.91 | 10 | 12 | 1 4 5 8 10 11 13 14 18 19 23 25 |
| F | ünsüz | 1686 | 1.81 | 14 | 2.23 | 11 | 14 | 1 2 3 7 8 12 14 15 16 17 20 21 23 24 |
| V | ünsüz | 1597 | 1.71 | 13 | 2.07 | 11 | 14 | 2 3 4 6 9 11 12 14 15 16 19 21 22 24 |
| EY | ünlü | 1585 | 1.70 | 12 | 1.91 | 10 | 12 | 1 5 6 7 10 13 14 16 17 19 23 24 |
| OW | ünlü | 1543 | 1.65 | 11 | 1.75 | 11 | 15 | 1 3 6 9 10 12 15 17 18 19 20 21 22 23 25 |
| AO | ünlü | 1469 | 1.57 | 9 | 1.43 | 9 | 11 | 2 5 6 8 11 14 15 18 19 21 23 |
| P | ünsüz | 1360 | 1.46 | 10 | 1.59 | 9 | 11 | 1 4 5 10 11 12 15 17 18 21 22 |
| ER | ünlü | 1312 | 1.41 | 14 | 2.23 | 10 | 14 | 2 3 5 7 10 12 15 17 19 20 22 23 24 25 |
| Y | ünsüz | 1193 | 1.28 | 13 | 2.07 | 12 | 12 | 2 4 7 8 9 11 12 13 14 16 17 20 |
| UH | ünlü | 1121 | 1.20 | 8 | 1.27 | 8 | 9 | 2 6 7 10 12 13 17 18 22 |
| G | ünsüz | 1100 | 1.18 | 7 | 1.11 | 7 | 9 | 1 5 6 8 15 16 19 22 25 |
| NG | ünsüz | 1087 | 1.16 | 9 | 1.43 | 8 | 11 | 1 4 5 9 12 13 16 17 21 24 25 |
| AA | ünlü | 871 | 0.93 | 7 | 1.11 | 6 | 8 | 3 7 8 13 16 18 23 25 |
| SH | ünsüz | 689 | 0.74 | 10 | 1.59 | 8 | 10 | 3 5 6 11 13 15 18 20 21 24 |
| AW | ünlü | 685 | 0.73 | 6 | 0.95 | 6 | 6 | 1 3 4 12 18 19 |
| TH | ünsüz | 645 | 0.69 | 9 | 1.43 | 9 | 12 | 1 4 5 9 12 13 14 18 19 21 24 25 |
| JH | ünsüz | 364 | 0.39 | 7 | 1.11 | 7 | 8 | 1 6 7 9 14 17 20 21 |
| CH | ünsüz | 343 | 0.37 | 7 | 1.11 | 7 | 10 | 2 3 4 11 14 16 19 22 23 24 |
| OY | ünlü | 56 | 0.06 | 3 | 0.48 | 3 | 4 | 3 9 11 21 |
| ZH | ünsüz | 23 | 0.02 | 5 | 0.79 | 5 | 6 | 4 8 10 15 20 22 |

### Fonem dağılımı

- Temel setteki fonem oranları veri setindeki oranlarla büyük ölçüde örtüşüyor (Pearson r = 0.93).
- Ünlü payı: veri setinde %40.0, temel sette %39.7.
- Veri setine göre en çok öne çıkarılan fonemler (nadir sesler, bilinçli olarak): ZH (%0.02 → %0.79), OY (%0.06 → %0.48), CH (%0.37 → %1.11), JH (%0.39 → %1.11), SH (%0.74 → %1.59).
- En çok geri kalan fonemler (sık sesler, yine de birçok cümlede var): W (%3.34 → %1.75), IY (%3.93 → %2.23), AY (%2.62 → %1.59), T (%8.45 → %5.72), M (%3.15 → %2.38).

## Yöntem

1. **Tekrarların çıkarılması:** Cümleler büyük harfe çevrilip noktalama silinerek karşılaştırıldı; 53 tekrar çıkarıldı.
2. **Fonemlerin belirlenmesi:** Veri setinin kendi referans transkripsiyonları (`resource/scores.json`, ARPAbet) kullanıldı; phonemizer/eSpeak kurulumu gerekmedi. Vurgu rakamları sayım için atıldı ve CMU sözlüğünün 39 fonemi kaldı. İki kaydında farklı transkripsiyon bulunan 29 cümlede bir fonem, yalnızca tüm transkripsiyonlarda geçtiği kadar sayıldı.
3. **Aday havuzu:** 4–10 kelimelik cümleler; elle çıkarılan 7 cümle hariç 4776 aday.
4. **Açgözlü seçim:** Her adımda şu sırayla en iyi cümle eklendi: (a) henüz kapsanmamış fonem sayısı, (b) her fonemin en az 3 temel cümlede geçmesine katkı, (c) seçimde az geçen ve veri setinde nadir olan fonemlere öncelik veren azalan getirili puan, (d) eşitlikte daha kısa cümle.
5. **Yedekler:** Aynı yöntemle, hedef fonem başına 4 cümleye çıkarılarak seçildi.
6. **Pilot alt kümesi:** Temel cümleler arasından aynı yöntemle en çok fonemi kapsayan 5 cümle.

## Elle çıkarılan cümleler

Cümleler Türk katılımcılara okutulacağı için (Adım 8), algoritmanın seçtiği cümleler tek tek okundu. Dilbilgisi bozuk ya da hassas içerikli (şiddet, ölüm, siyaset) olanlar `resources/excluded_sentences.csv` listesine eklenip seçim yeniden çalıştırıldı.

| Cümle | Gerekçe |
| --- | --- |
| HE CALLED FOR EDUCATION CAMPAIGN TARGETING THE ISSUE OF WEAPONS | dilbilgisi: tanımlık eksik (an education campaign); silah konusu |
| NEARLY ALL REPUBLICANS VOTED AGAINST THE MEASURE | siyasi içerik |
| GUNS WERE USED FOR ABOUT THREE OF EVERY FIVE KILLINGS | şiddet içeriği |
| TAKE HERE AND NO SMALL PART SHOULD BE THEIR | anlamsız/bozuk cümle |
| DEATH SHOULD BE VEILED THE GRAVE SHOULD HAVE ITS RESERVE | ölüm konusu; eski edebi dil |
| TINA LIFE IN NEW USUAL | anlamsız/bozuk cümle |
| THIS ELECTION IS GOING TO BE A TURNING POINT | siyasi içerik (seçim) |

## Sınırlılıklar

- **Referans aksan karışık:** Transkripsiyonlar kayıt bazında yazılmış ve tek bir aksana bağlı değil. Örneğin "FOR" 278 kayıtta `F AO` (r'siz, İngiliz), 40 kayıtta `F AO R` (Amerikan). R ve ER sayıları okuyan kişinin aksanına göre değişebilir. Bu fark Adım 4'te ele alınıyor.
- **Her cümlenin tek kaynak kaydı var.** Cümlelerin fonemleri o kaydın transkripsiyonuna dayanıyor.
- **Cümle uygunluğu elle değerlendirildi.** Bazı cümleler edebi bir dile sahip (ör. "YET WHY THERE SHOULD BE NOTHING IS HARD TO SAY"); başlangıç düzeyi için zorluğu pilot kayıtlarda gözlenmeli.
- **Fonem çiftleri (ses geçişleri) kapsamı ölçülmedi;** seçim tek tek fonemlere göre yapıldı.
