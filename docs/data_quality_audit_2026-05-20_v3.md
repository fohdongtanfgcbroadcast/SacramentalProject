# 데이터 품질 감사 v3 — vermigli 소스 교체 후 (2026-05-20)

## 배경

이 보고서는 동일 2026-05-20 일자의 v2(`data_quality_audit_2026-05-20.md`, 커밋 `cc820da`) 이후 **vermigli 컬렉션의 소스를 EEBO-TCP A14350(TEI P5, CC0 1.0)으로 교체하고 재인제스트한 결과**를 측정한 v3 이다.

v2 시점에서 vermigli 가 새 상위 오염 1위(junk 4.2%)였고, OCR Class C(블랙레터 djvu 깨짐, strip 무효) 로 분류돼 별도 결정 대기 상태였다. EEBO-TCP 가 사람이 직접 keyboarded 한 동일 1583 Marten 영역본의 TEI P5 transcription 을 CC0 1.0 으로 제공한다는 사실을 확인하여, djvu OCR 두 권(common_places_vol{1,2}.txt) 을 단일 TEI 추출본(common_places.txt)으로 교체했다.

추출 도구: `scripts/extract_eebo_tcp.py` (TEI body → plain text, long-s `ſ/ʃ` → `s` 정규화, `<note>` 별도 줄, `<gap>` → '…', Early Modern English 표기는 보존).

## v2 → v3 변화 — vermigli 소스 교체 효과

| 지표 | v2 (djvu OCR) | v3 (EEBO-TCP TEI) | 변화 |
|---|--:|--:|---|
| junk % | **4.2** | **0.0** | **-4.2pp** |
| severe % | 0.0 | 0.0 | — |
| ocr % | 2.5 | 6.5 | +4.0pp* |
| 청크 수 | 21,507 | 21,593 | +86 |

\* ocr % 증가는 손상이 아니라 정상 라틴어 인용·구절번호·각주가 보존된 결과(EEBO-TCP 가 `<foreign>`/`<note>` 를 보존하기 때문). step 2 의 strip 범위 밖이며 검색 품질에 무해.

**별도 측정 — 원본 단위 OCR 깨짐 시그니처(자음 5+ 연속 단어):**

| 소스 | 자음 5+ 연속 단어 |
|---|--:|
| djvu OCR (vol1) | 43,955 |
| djvu OCR (vol2) | 13,711 |
| **djvu OCR 합계** | **57,666** |
| **EEBO-TCP TEI(common_places.txt)** | **372** (-99.4%) |

예시:
- djvu: `publtfk`, `Antichnftian`, `makfigmcntionf`, `lndttmoifeth`, `srssss`, `tmmmmmt`
- TEI: `Martyrs`, `martyrs`, `Apocrypha`, `lightly`, `witchcraft`, `nightstealing` — 모두 정상 영어 단어

→ vermigli 컬렉션은 이제 **검색·임베딩 품질의 노이즈 원인에서 제거**됐다. 새 상위 1위는 bunyan 4.0%(Gutenberg license boilerplate, Class A — 추가 strip 규칙으로 해소 가능 범위).

> **ocr % 한글 소스 오탐 경고**: OCR 휴리스틱은 알파벳을 `[A-Za-z]` 로만 센다. 한글 소스 컬렉션은 한글 글자가 전부 비알파로 계산돼 ocr % 가 비정상적으로 부풀려진다. `moltmann` ocr 99.5% 는 OCR 깨짐이 아니라 한글 본문 — 무시할 것.

## 측정 방법

- 전수 스캔: 58 컬렉션 / 총 599,521 청크 (샘플링 아님)
- **junk %**: `_IA_JUNK`/`_JUNK_RE` 매칭 또는 `_NAV_LABELS` 포함 라인이 1개 이상인 청크 비율 (= step 2 가 실제 제거할 대상). 측정 심볼은 `translate_confessions` 에서 직접 import — 정의 일치 보장
- **severe %**: `_drop_boilerplate_lines` 적용 시 길이가 30% 초과 줄어드는 청크 비율 (심한 오염)
- **평균 축소**: 전 청크 평균 `_drop_boilerplate_lines` 축소율
- **ocr %**(별도 축, **step 2 범위 외**): 비알파 비율 50% 초과 라인 포함 청크. strip 로직이 다루지 않음 — 보존처리/별도 작업 대상

> 주의: 청크 오버랩 200자로 경계 잡음이 두 청크에 중복 출현할 수 있다. "오염 청크 비율"로는 정직한 수치다(어느 사본이 검색돼도 잡음 노출). 중복 제거는 하지 않는다.

## 컬렉션별 오염도 (junk % 내림차순)

| 컬렉션 | 청크 | junk % | severe % | ocr % | 평균축소 % |
|---|--:|--:|--:|--:|--:|
| bunyan | 2,538 | 4.0 | 1.7 | 1.1 | 1.1 |
| confessions | 6,470 | 3.4 | 1.3 | 2.1 | 1.1 |
| kierkegaard | 1,040 | 3.3 | 1.1 | 0.7 | 0.9 |
| edwards | 4,032 | 3.2 | 1.4 | 2.2 | 1.0 |
| luther | 14,941 | 2.6 | 1.0 | 2.9 | 0.7 |
| harnack | 5,841 | 2.1 | 0.6 | 4.0 | 0.4 |
| melanchthon | 1,541 | 2.1 | 0.9 | 4.3 | 0.6 |
| schweitzer | 3,253 | 2.1 | 0.7 | 4.9 | 0.5 |
| whitefield | 9,934 | 1.7 | 0.7 | 9.4 | 0.5 |
| bernard | 2,461 | 1.6 | 0.6 | 17.5 | 0.4 |
| anselm | 2,715 | 1.4 | 0.6 | 20.4 | 0.4 |
| francis | 3,009 | 1.3 | 0.4 | 1.7 | 0.3 |
| john_damascus | 6,860 | 1.0 | 0.4 | 8.3 | 0.3 |
| basil | 8,048 | 1.0 | 0.0 | 4.3 | 0.0 |
| baxter | 18,586 | 0.9 | 0.3 | 2.6 | 0.3 |
| knox | 8,223 | 0.9 | 0.4 | 5.6 | 0.2 |
| rutherford | 3,872 | 0.9 | 0.4 | 5.4 | 0.2 |
| ambrose | 5,182 | 0.7 | 0.0 | 8.2 | 0.0 |
| julian_norwich | 760 | 0.5 | 0.0 | 28.8 | 0.0 |
| eckhart | 771 | 0.5 | 0.0 | 11.2 | 0.0 |
| aquinas | 40,741 | 0.4 | 0.2 | 19.8 | 0.1 |
| bonaventure | 5,165 | 0.4 | 0.0 | 5.7 | 0.0 |
| bullinger | 10,585 | 0.4 | 0.1 | 9.1 | 0.1 |
| scotus | 1,375 | 0.4 | 0.0 | 7.1 | 0.0 |
| athanasius | 26,443 | 0.3 | 0.1 | 7.5 | 0.1 |
| mather | 14,368 | 0.3 | 0.1 | 1.4 | 0.1 |
| kempis | 762 | 0.3 | 0.0 | 26.8 | 0.0 |
| law | 3,831 | 0.3 | 0.0 | 2.9 | 0.0 |
| hodge | 13,506 | 0.3 | 0.1 | 2.9 | 0.1 |
| calvin | 131,074 | 0.2 | 0.0 | 13.2 | 0.0 |
| forsyth | 2,386 | 0.2 | 0.0 | 0.0 | 0.0 |
| murray | 963 | 0.2 | 0.0 | 7.6 | 0.0 |
| bavinck | 1,957 | 0.2 | 0.0 | 4.0 | 0.0 |
| zwingli | 4,116 | 0.2 | 0.0 | 0.7 | 0.0 |
| rauschenbusch | 2,658 | 0.2 | 0.0 | 2.5 | 0.0 |
| augustine | 63,593 | 0.2 | 0.1 | 19.0 | 0.0 |
| cyril_jerusalem | 1,730 | 0.2 | 0.0 | 5.5 | 0.0 |
| moltmann | 12,114 | 0.2 | 0.0 | 99.5 | 0.0 |
| chrysostom | 41,411 | 0.1 | 0.0 | 9.3 | 0.0 |
| schleiermacher | 9,054 | 0.1 | 0.0 | 2.8 | 0.0 |
| gregory_i | 16,380 | 0.1 | 0.0 | 14.2 | 0.0 |
| tertullian | 7,207 | 0.1 | 0.0 | 10.1 | 0.0 |
| clement_alexandria | 4,511 | 0.1 | 0.0 | 10.7 | 0.0 |
| celano | 1,319 | 0.1 | 0.0 | 4.9 | 0.0 |
| kuyper | 4,126 | 0.1 | 0.0 | 1.0 | 0.0 |
| cyprian | 4,606 | 0.1 | 0.0 | 10.1 | 0.0 |
| owen | 4,862 | 0.1 | 0.0 | 9.0 | 0.0 |
| ritschl | 3,252 | 0.1 | 0.0 | 1.8 | 0.0 |
| origen | 5,211 | 0.1 | 0.0 | 6.8 | 0.0 |
| irenaeus | 4,745 | 0.0 | 0.0 | 8.5 | 0.0 |
| cranmer | 12,023 | 0.0 | 0.0 | 5.0 | 0.0 |
| justin_martyr | 1,977 | 0.0 | 0.0 | 3.1 | 0.0 |
| watts | 1,215 | 0.0 | 0.0 | 11.5 | 0.0 |
| zinzendorf | 1,184 | 0.0 | 0.0 | 10.2 | 0.0 |
| jerome | 5,234 | 0.0 | 0.0 | 2.8 | 0.0 |
| vermigli | 21,593 | 0.0 | 0.0 | 6.5 | 0.0 |
| benedict | 246 | 0.0 | 0.0 | 17.1 | 0.0 |
| wesley | 1,921 | 0.0 | 0.0 | 0.9 | 0.0 |

## 상위 3개 오염 컬렉션 — 잡음 청크 표본 (원문 그대로)


### bunyan (junk 4.0%)

**표본 1:**
```
﻿The Project Gutenberg eBook of The Pilgrim's Progress from this world to that which is to come
 
This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this eBook or online
at www.gutenberg.org. If you are not located in the United States,
you will have to check the laws of the country where you are located
before using this eBook.

Title: The Pilgrim's Progress from this world to that whic…
```

**표본 2:**
```
eans that no one owns a United States copyright in these works,
so the Foundation (and you!) can copy and distribute it in the United
States without permission and without paying copyright
royalties. Special rules, set forth in the General Terms of Use part
of this license, apply to copying and distributing Project
Gutenberg™ electronic works to protect the PROJECT GUTENBERG™
concept and trademark. Project Gutenberg is a registered trademark,
and may not be used if you charge for an eBook, except by following
the terms of the trademark license, including paying royalties for use
of the Project…
```

**표본 3:**
```
he trademark license, including paying royalties for use
of the Project Gutenberg trademark. If you do not charge anything for
copies of this eBook, complying with the trademark license is very
easy. You may use this eBook for nearly any purpose such as creation
of derivative works, reports, performances and research. Project
Gutenberg eBooks may be modified and printed and given away—you may
do practically ANYTHING in the United States with eBooks not protected
by U.S. copyright law. Redistribution is subject to the trademark
license, especially commercial redistribution.

START: FULL LICENSE…
```


### confessions (junk 3.4%)

**표본 1:**
```
and the third, the resurrection of the dead; yet not of all, but as it is said: The Lord shall come and all His saints with Him. Then shall the world see the Lord coming upon the clouds of heaven. 

 
About this page
 
Source. Translated by M.B. Riddle. From Ante-Nicene Fathers, Vol. 7. Edited by Alexander Roberts, James Donaldson, and A. Cleveland Coxe. (Buffalo, NY: Christian Literature Publishing Co., 1886.) Revised and edited for New Advent by Kevin Knight. <https://www.newadvent.org/fathers/0714.htm>.
 

Contact information. The editor of New Advent is Kevin Knight. My email address is fe…
```

**표본 2:**
```
﻿The Project Gutenberg eBook of Luther's Little Instruction Book: The Small Catechism of Martin Luther
 
This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this eBook or online
at www.gutenberg.org. If you are not located in the United States,
you will have to check the laws of the country where you are located
before using this eBook.

Title: Luther's Little Instruction Book: The Small …
```

**표본 3:**
```
mith. It has
been placed in the public domain by him. You may freely distribute,
copy or print this text. Please direct any comments or suggestions to
Rev. Robert E. Smith of the Walther Library at:

Concordia Theological Seminary
Email: smithre@mail.ctsfw.edu
Surface Mail: 6600 N. Clinton St., Ft. Wayne, IN 46825
USA Phone: (260) 452-3149 Fax: (260) 452-2126

*** END OF THE PROJECT GUTENBERG EBOOK LUTHER
```


### kierkegaard (junk 3.3%)

**표본 1:**
```
﻿The Project Gutenberg eBook of Selections from the Writings of Kierkegaard
 
This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this eBook or online
at www.gutenberg.org. If you are not located in the United States,
you will have to check the laws of the country where you are located
before using this eBook.

Title: Selections from the Writings of Kierkegaard

Author: Søren Kierkegaard
…
```

**표본 2:**
```
om the Writings of Kierkegaard

Author: Søren Kierkegaard

Translator: Lee M. Hollander

 
Release date: September 20, 2019 [eBook #60333]
 Most recently updated: October 17, 2024

Language: English

Other information and formats: www.gutenberg.org/ebooks/60333

Credits: Produced by Laura Natal Rodrigues at Free Literature (Images
 generously made available by Internet Archive.)

UNIVERSITY OF TEXAS BULLETIN

NO. 2326: JULY 8, 1923

SELECTIONS FROM THE WRITINGS OF KIERKEGAARD

TRANSLATED BY L. M. HOLLANDER

ADJUNCT PROFESSOR OF GERMANIC LANGUAGES

COMPARATIVE LITERATURE SERIES NO. 3

PUBLISHED…
```

**표본 3:**
```
ED BY THE UNIVERSITY OF TEXAS, AUSTIN

The benefits of education and of
useful knowledge, generally diffused
through a community, are essential
to the preservation of a free government.

Sam Houston

Cultivated mind is the guardian
genius of democracy.... It is the
only dictator that freemen acknowledge
and the only security that free-men
desire.

Mirabeau B. Lamar

_To my Father-in-Law
The Reverend George Fisher,
A Christian._

[Illustration 01]

[Illustration 02]

CONTENTS
INTRODUCTION.
DIAPSALMATA.
IN VINO VERITAS (THE BANQUET).
FEAR AND TREMBLING.
PREPARATION FOR A CHRISTIAN LIFE.
THE PRES…
```


## 다음 단계 (결정 보류 — 측정만 보고)

본 보고서는 raw 측정치만 제시한다. 어떤 컬렉션을 재인제스트할지(임계값·범위)는 step 2 에서 사용자와 결정한다.

