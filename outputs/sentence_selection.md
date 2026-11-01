# Okutulacak cümlelerin seçimi

Kaynak: speechocean762, 5000 kayıt → 4947 tekrarsız cümle (53 tekrar çıkarıldı). Aday havuzu: 4–10 kelimelik, elle çıkarılan 7 cümle hariç 4776 cümle.

Fonemler veri setinin kendi referans transkripsiyonlarından (ARPAbet) alındı; vurgu rakamları sayım için atıldı ve CMU sözlüğünün 39 fonemi kaldı. Aynı cümlenin iki kaydında farklı transkripsiyon olan 29 cümlede bir fonem, yalnızca tüm transkripsiyonlarda geçtiği kadar sayıldı.

Seçim açgözlü algoritmayla yapıldı: önce henüz kapsanmayan fonemler, ardından her fonemin en az 3 temel cümlede geçmesi, ardından seçimde az geçen ve veri setinde nadir olan fonemlere öncelik veren azalan getirili bir puan; eşitlikte daha kısa cümle. Yedek cümleler aynı yöntemle hedef 4 cümleye çıkarılarak seçildi.

## Özet

- Temel 20 cümle 39 fonemin **39**'unu kapsıyor.
- En az geçen fonem: **OY**, 3 temel cümlede.
- Her fonem en az 3 temel cümlede, yedeklerle birlikte en az 4 cümlede geçiyor.
- Temel set: 629 fonem, 182 kelime.
- Pilot (5 temel cümle): 39 fonemin 39'unu kapsıyor.

## Cümleler

| # | Rol | Pilot | Cümle | Kelime | Yeni kapsanan fonemler | Kaynak kayıtlar | Ort. total puanı |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | temel | 1 | AMERICA'S GOING THROUGH JUST ABOUT THE SAME EXPERIENCE AS MYSELF | 10 | AE AH AW AY B DH EH EY F G IH JH K L M N NG OW P R S T TH UW Z | 090880095 (test) | 8.0 |
| 2 | temel | 2 | DO YOU HAVE ANY QUESTIONS FOR HER | 7 | AO CH D ER HH IY UH V W Y | 012930139 (test) | 8.0 |
| 3 | temel | 3 | HOWEVER MARKET CONDITIONS HAVE LEFT US WITH NO OTHER CHOICE | 10 | AA OY SH | 007650036 (test) | 6.0 |
| 4 | temel | 4 | THERE'S NOTHING THAT UNUSUAL ABOUT THE STRUCTURE OF THE PLAN | 10 | ZH | 022360321 (train) | 7.0 |
| 5 | temel |  | THEY WERE PROBABLY DATING OR SOMETHING HE SHRUGGED | 8 | — | 096470012 (test) | 3.0 |
| 6 | temel |  | HE SHOULD HAVE LET THE ORGANIZATION AND THE MANAGER KNOW | 10 | — | 023800114 (train) | 8.0 |
| 7 | temel |  | I JUST WANT TO SEE HER SAFE IN YOUR HANDS | 10 | — | 096180018 (test) | 6.0 |
| 8 | temel |  | THIS IS NOT AN ARGUMENT FOR BUSINESS AS USUAL | 9 | — | 029370106 (test) | 8.0 |
| 9 | temel |  | SO ENJOY AND TELL ME WHAT YOU THINK OF IT | 10 | — | 096250021 (train) | 3.0 |
| 10 | temel |  | BUT THAT WOULD NOT ALONE EXPLAIN HIS PLEASURE | 8 | — | 095530066 (test) | 8.0 |
| 11 | temel |  | YOUR SECOND SHOT WILL BE A CHOICE OF TWO OPTIONS | 10 | — | 004820114 (train) | 8.0 |
| 12 | temel |  | THE MOST POWERFUL THING YOU CAN DO IS RECOVER | 9 | — | 023550201 (test) | 9.0 |
| 13 | temel | 5 | YET WHY THERE SHOULD BE NOTHING IS HARD TO SAY | 10 | — | 081530100 (test) | 8.0 |
| 14 | temel |  | YOU HAVE TO CHANGE WITH THEM OR BE LEFT BEHIND | 10 | — | 096080027 (test) | 4.0 |
| 15 | temel |  | I SHALL NEVER FORGIVE MYSELF FOR THIS EXPOSURE | 8 | — | 004820378 (train) | 8.0 |
| 16 | temel |  | YOU HAVE DONE THAT WHICH MUST MAKE A FATHER ANGRY | 10 | — | 054080081 (train) | 8.0 |
| 17 | temel |  | THE SOLDIER STARED INTO SPACE MIND RACING FURIOUSLY | 8 | — | 096450011 (train) | 4.0 |
| 18 | temel |  | SHE OPENED THE DOOR AND LOOKED AROUND THE BATHROOM | 9 | — | 096000018 (train) | 5.0 |
| 19 | temel |  | BOTH GIRLS CALLED OUT AS THEY WATCHED HER LEAVE | 9 | — | 096470005 (test) | 4.0 |
| 20 | temel |  | HER EYES USUALLY SO GENTLE FLASHED FIRE | 7 | — | 096400016 (test) | 4.0 |
| 21 | yedek |  | THANKS FOR THE JOY OF OWNERSHIP | 6 | — | 020590121 (train) | 8.0 |
| 22 | yedek |  | COME IT WOULD GIVE ME SO MUCH PLEASURE | 8 | — | 091010151 (test) | 9.0 |
| 23 | yedek |  | I AM RICH AND SO WAS MY FATHER BEFORE ME | 10 | — | 096040023 (train) | 4.0 |
| 24 | yedek |  | I THINK THE SHAKING HURT HIS FEELINGS VERY MUCH | 9 | — | 052180112 (train) | 8.0 |
| 25 | yedek |  | ANGER IS SOMETHING THAT CAN BE REALLY HARD TO CONTROL | 10 | — | 023550230 (test) | 9.0 |

## Elle çıkarılan cümleler

Cümleler çalışma katılımcılarına okutulacağı için dilbilgisi bozuk ya da hassas içerikli (şiddet, ölüm, siyaset) olanlar `resources/excluded_sentences.csv` listesiyle aday havuzundan çıkarıldı.

| Cümle | Gerekçe |
| --- | --- |
| HE CALLED FOR EDUCATION CAMPAIGN TARGETING THE ISSUE OF WEAPONS | dilbilgisi: tanımlık eksik (an education campaign); silah konusu |
| NEARLY ALL REPUBLICANS VOTED AGAINST THE MEASURE | siyasi içerik |
| GUNS WERE USED FOR ABOUT THREE OF EVERY FIVE KILLINGS | şiddet içeriği |
| TAKE HERE AND NO SMALL PART SHOULD BE THEIR | anlamsız/bozuk cümle |
| DEATH SHOULD BE VEILED THE GRAVE SHOULD HAVE ITS RESERVE | ölüm konusu; eski edebi dil |
| TINA LIFE IN NEW USUAL | anlamsız/bozuk cümle |
| THIS ELECTION IS GOING TO BE A TURNING POINT | siyasi içerik (seçim) |

## Fonem kapsama tablosu

Sıralama: tüm veri setindeki sıklık. Oranlar fonem sayısına göre; "cümle" sütunları fonemin kaç cümlede geçtiğini gösterir.

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

## Not: referans aksan

Referans transkripsiyonlar kayıt bazında yazılmış ve tek bir aksana bağlı değil. Örneğin "FOR" 278 kayıtta `F AO` (r'siz, İngiliz), 40 kayıtta `F AO R` (Amerikan) olarak geçiyor. Bu nedenle özellikle R ve ER fonemlerinin sayıları, cümleyi okuyacak kişinin aksanına göre değişebilir.
