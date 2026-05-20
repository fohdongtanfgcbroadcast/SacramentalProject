# 데이터 품질 감사 v7 — Class A 2차(11개) 일괄 재인제스트 후 (2026-05-21)

## 배경

이 보고서는 v6(`data_quality_audit_2026-05-21.md`, 커밋 `a467c30` — Class A 1차 7개 후) 이후 **Class A 2차 11개(junk ≥ 0.5%) 를 일괄 재인제스트한 결과**를 측정한 v7 다.

대상은 사용자 임계값 결정으로 정해졌다: junk ≥ 0.5% 11개(whitefield · anselm · francis · john_damascus · basil · baxter · knox · rutherford · ambrose · eckhart · julian_norwich) — "한 번에 완전 정리" 옵션.

## v6 → v7 변화 — Class A 2차 11개 효과

| 컬렉션 | v6 청크 | v7 청크 | v6 junk % | v7 junk % | Δ |
|---|--:|--:|--:|--:|--:|
| whitefield | 9,934 | 9,762 | 1.7 | **0.0** | -1.7pp |
| anselm | 2,715 | 2,675 | 1.4 | **0.0** | -1.4pp |
| francis | 3,009 | 2,971 | 1.3 | **0.0** | -1.3pp |
| john_damascus | 6,860 | 6,788 | 1.0 | **0.0** | -1.0pp |
| basil | 8,048 | 8,050 | 1.0 | **0.0** | -1.0pp |
| baxter | 18,586 | 18,426 | 0.9 | **0.0** | -0.9pp |
| knox | 8,223 | 8,135 | 0.9 | **0.0** | -0.9pp |
| rutherford | 3,872 | 3,813 | 0.9 | **0.0** | -0.9pp |
| ambrose | 5,182 | 5,185 | 0.7 | **0.0** | -0.7pp |
| eckhart | 771 | 770 | 0.5 | **0.0** | -0.5pp |
| julian_norwich | 760 | 759 | 0.5 | **0.0** | -0.5pp |

**11개 모두 정확히 0.0% 로 해소.** 총 청크 607,137 → 606,511 (-626, strip 자연 감소). 코드 변경 없이 일괄 재인제스트만 적용.

## 새 상위 오염 — bernard 1.6% (별도 패턴 발견)

v7 시점 새 상위 1위는 **bernard 1.6%** (직전 v6 측정에 없던 발견). 표본 분석 결과 strip 이 잡지 못하는 **CCEL header/footer 메타데이터 블록** 패턴:

```
__________________________________________________________________
 Title: On Loving God
 Creator(s): Bernard, of Clairvaux, Saint (1090 or 91-1153)
 CCEL Subjects: All; Classic; Christian Life; Proofed
 LC Call no: BV4817
...
__________________________________________________________________
```

그리고 본문 끝의 "References" + `file:///ccel/.../cache/...html` URL 리스트. `_JUNK_RE` 의 `Christian Classics Ethereal Library` 가 청크에 포함되면 매칭 잡혀 junk 로 카운트되지만, "Title:" / "Creator(s):" / `file:///...` 등 *대다수 라인* 은 매치 안 됨. 이건 **strip 코드 보강 후보** (CCEL 헤더/푸터 블록 슬라이싱 규칙 추가).

aquinas 0.4% 도 동일 패턴(Gutenberg 헤더 메타 라인 + ELECTRONIC EDITION 노트 블록). Class A 잔여 임계값 컷오프 아래라 이번에 누락된 잔여.

> **ocr % 한글 소스 오탐 경고**: `moltmann` ocr 99.5% 는 OCR 깨짐이 아니라 한글 본문 — 무시할 것.

## 측정 방법

- 전수 스캔: 58 컬렉션 / 총 606,511 청크 (샘플링 아님)
- **junk %**: `_IA_JUNK`/`_JUNK_RE` 매칭 또는 `_NAV_LABELS` 포함 라인이 1개 이상인 청크 비율 (= step 2 가 실제 제거할 대상). 측정 심볼은 `translate_confessions` 에서 직접 import — 정의 일치 보장
- **severe %**: `_drop_boilerplate_lines` 적용 시 길이가 30% 초과 줄어드는 청크 비율 (심한 오염)
- **평균 축소**: 전 청크 평균 `_drop_boilerplate_lines` 축소율
- **ocr %**(별도 축, **step 2 범위 외**): 비알파 비율 50% 초과 라인 포함 청크. strip 로직이 다루지 않음 — 보존처리/별도 작업 대상

> 주의: 청크 오버랩 200자로 경계 잡음이 두 청크에 중복 출현할 수 있다. "오염 청크 비율"로는 정직한 수치다(어느 사본이 검색돼도 잡음 노출). 중복 제거는 하지 않는다.

## 컬렉션별 오염도 (junk % 내림차순)

| 컬렉션 | 청크 | junk % | severe % | ocr % | 평균축소 % |
|---|--:|--:|--:|--:|--:|
| bernard | 2,461 | 1.6 | 0.6 | 17.5 | 0.4 |
| aquinas | 40,741 | 0.4 | 0.2 | 19.8 | 0.1 |
| bonaventure | 5,165 | 0.4 | 0.0 | 5.7 | 0.0 |
| bullinger | 10,585 | 0.4 | 0.1 | 9.1 | 0.1 |
| scotus | 1,375 | 0.4 | 0.0 | 7.1 | 0.0 |
| athanasius | 26,443 | 0.3 | 0.1 | 7.5 | 0.1 |
| mather | 14,368 | 0.3 | 0.1 | 1.4 | 0.1 |
| kempis | 762 | 0.3 | 0.0 | 26.8 | 0.0 |
| law | 3,831 | 0.3 | 0.0 | 2.9 | 0.0 |
| hodge | 13,506 | 0.3 | 0.1 | 2.9 | 0.1 |
| confessions | 4,776 | 0.3 | 0.0 | 2.5 | 0.0 |
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
| schweitzer | 3,193 | 0.1 | 0.0 | 4.9 | 0.0 |
| owen | 4,862 | 0.1 | 0.0 | 9.0 | 0.0 |
| ritschl | 3,252 | 0.1 | 0.0 | 1.8 | 0.0 |
| origen | 5,211 | 0.1 | 0.0 | 6.8 | 0.0 |
| harnack | 5,530 | 0.1 | 0.0 | 1.9 | 0.0 |
| knox | 8,135 | 0.0 | 0.0 | 5.5 | 0.0 |
| irenaeus | 4,745 | 0.0 | 0.0 | 8.5 | 0.0 |
| luther | 14,433 | 0.0 | 0.0 | 2.9 | 0.0 |
| cranmer | 12,023 | 0.0 | 0.0 | 5.0 | 0.0 |
| basil | 8,050 | 0.0 | 0.0 | 4.4 | 0.0 |
| ambrose | 5,185 | 0.0 | 0.0 | 8.3 | 0.0 |
| john_damascus | 6,788 | 0.0 | 0.0 | 8.4 | 0.0 |
| baxter | 18,426 | 0.0 | 0.0 | 2.7 | 0.0 |
| wesley | 12,407 | 0.0 | 0.0 | 8.7 | 0.0 |
| justin_martyr | 1,977 | 0.0 | 0.0 | 3.1 | 0.0 |
| whitefield | 9,762 | 0.0 | 0.0 | 9.4 | 0.0 |
| rutherford | 3,813 | 0.0 | 0.0 | 5.3 | 0.0 |
| watts | 1,215 | 0.0 | 0.0 | 11.5 | 0.0 |
| francis | 2,971 | 0.0 | 0.0 | 1.7 | 0.0 |
| zinzendorf | 1,184 | 0.0 | 0.0 | 10.2 | 0.0 |
| julian_norwich | 759 | 0.0 | 0.0 | 28.9 | 0.0 |
| kierkegaard | 1,006 | 0.0 | 0.0 | 0.9 | 0.0 |
| eckhart | 770 | 0.0 | 0.0 | 11.0 | 0.0 |
| jerome | 5,234 | 0.0 | 0.0 | 2.8 | 0.0 |
| edwards | 3,919 | 0.0 | 0.0 | 1.8 | 0.0 |
| bunyan | 2,426 | 0.0 | 0.0 | 1.2 | 0.0 |
| anselm | 2,675 | 0.0 | 0.0 | 20.5 | 0.0 |
| vermigli | 21,593 | 0.0 | 0.0 | 6.5 | 0.0 |
| melanchthon | 1,503 | 0.0 | 0.0 | 4.4 | 0.0 |
| benedict | 246 | 0.0 | 0.0 | 17.1 | 0.0 |

## 상위 3개 오염 컬렉션 — 잡음 청크 표본 (원문 그대로)


### bernard (junk 1.6%)

**표본 1:**
```
__________________________________________________________________

 Title: On Loving God
 Creator(s): Bernard, of Clairvaux, Saint (1090 or 91-1153)
 CCEL Subjects: All; Classic; Christian Life; Proofed
 LC Call no: BV4817
 LC Subjects:

 Practical theology

 Practical religion. The Christian life

 Works of meditation and devotion
 __________________________________________________________________

 ON LOVING GOD

by St. Bernard of Clairvaux

 Made available to the net by Paul Halsall <HALSALL@MURRAY.FORDHAM.EDU>.

 __________________________________________________________________

 DEDICAT…
```

**표본 2:**
```
[124]15:46 [125]15:46 [126]15:50

 [127]4:17 [128]5:9 [129]5:16

 Galatians

 [130]4:4

 Ephesians

 [131]5:27 [132]5:29

 Philippians

 [133]2:10 [134]2:21 [135]3:20

 Colossians

 [136]3:5

 [137]5:21

 [138]1:5 [139]1:9 [140]6:8 [141]6:9

 [142]2:12

 Hebrews

 [143]9:12

 James

 [144]1:5 [145]1:14

 [146]1:22

 [147]3:18 [148]4:8 [149]4:18 [150]4:19

 Revelation

 [151]19:9 [152]21:5

 Wisdom of Solomon

 [153]9:15

 Sirach

 [154]18:30 [155]24:20 [156]24:21
 __________________________________________________________________

 This document is from the Christian Classics Ethereal
 Library…
```

**표본 3:**
```
__________________________________________

 This document is from the Christian Classics Ethereal
 Library at Calvin College, http://www.ccel.org,
 generated on demand from ThML source.

References

 1. file:///ccel/b/bernard/loving_god/cache/loving_god.html3#iv-p6.3
 2. file:///ccel/b/bernard/loving_god/cache/loving_god.html3#iv-p6.6
 3. file:///ccel/b/bernard/loving_god/cache/loving_god.html3#v-p4.1
 4. file:///ccel/b/bernard/loving_god/cache/loving_god.html3#ix-p6.2
 5. file:///ccel/b/bernard/loving_god/cache/loving_god.html3#vii-p2.2
 6. file:///ccel/b/bernard/loving_god/cache/loving_god.…
```


### aquinas (junk 0.4%)

**표본 1:**
```
﻿The Project Gutenberg eBook of Summa Theologica, Part I (Prima Pars)
 
This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this eBook or online
at www.gutenberg.org. If you are not located in the United States,
you will have to check the laws of the country where you are located
before using this eBook.

Title: Summa Theologica, Part I (Prima Pars)

Author: Saint Aquinas Thomas

 
Releas…
```

**표본 2:**
```
lish Dominican Province

BENZIGER BROTHERS
NEW YORK
_______________________

DEDICATION

To the Blessed Virgin
Mary Immaculate
Seat of Wisdom
_______________________

NOTE TO THIS ELECTRONIC EDITION

The text of this electronic edition was originally produced by Sandra
K. Perry, Perrysburg, Ohio, and made available through the Christian
Classics Ethereal Library <http://www.ccel.org>. I have eliminated
unnecessary formatting in the text, corrected some errors in
transcription, and added the dedication, tables of contents,
Prologue, and the numbers of the questions and articles, as they
appeare…
```

**표본 3:**
```
itations to books other than the Bible.

* Any matter that appeared in a footnote in the Benziger Brothers
edition is presented in brackets at the point in the text where the
footnote mark appeared.

* Greek words are presented in Roman transliteration.

* Paragraphs are not indented and are separated by blank lines.

* Numbered topics, set forth at the beginning of each question and
at certain other places, are ordinarily presented on a separate line
for each topic.

* Titles of questions are in all caps.

Anything else in this electronic edition that does not correspond to
the content of the…
```


### bonaventure (junk 0.4%)

**표본 1:**
```
r
 thee" [II Cor. 12:9]; let us exult with David, saying, "For Thee my
 flesh and my heart hath fainted away; Thou art the God of my heart, and
 the God that is my portion forever [Ps. 72, 26]. . . . Blessed be the
 Lord God of Israel from everlasting to everlasting; and let all the
 people say: So be it, so be it" [Ps. 105:48]. AMEN.
 __________________________________________________________________

 [11] "Mystic Theology," Ch. I [Migne, "Pat. Graec.," Vol. III, 997].

 [12] "Ibid."
 __________________________________________________________________

 This document is from the Christian Cla…
```

**표본 2:**
```
le . com/ 

• ■ +1 

Google 

Google 

THE 
TEMPLE 
CLASSICS 

THE 

LIFE OF SAINT FRANCIS 

BY 

SAINT BONAVENTURA 

Digitized by CjOOQle 

Published under the Auspices or 
The International Society of Franciscan Studies 
(British Branch) 

Digitized by L,OOQle 

Google 

firm,*, fC*> ^ 

Google 

The LIFE 
ySAINT 
FRANCIS 

SAINTS 

BONAVEN- 

TURA^t 

im 

CCCIV PU3LI5HGI; • BY ^ M DtN' 

Google 

AMXCCC1V PUBL15H6D • BY >M D6NT 
AMP CO: JSLOlNCHOUSe LOMDOM W 

S-rs 

304833 

• • < 

Google 

CONTENTS 

CHAP. PAGE 

Prologue . . i 

i. Of his Manner of Life in the Secular 

State . ' . . 7…
```

**표본 3:**
```
Needs . . 68 
viii. Of the Kindly Impulses of his Piety, and 
of how the Creatures lacking Under- 
standing seemed to be made Subject unto 
him . . ... 80 

ix. Of his Ardent Love, and Yearning for 

Martyrdom . . 04 

x. Of his Zeal and Efficacy in Prayer . 104 
xi. Of his Understanding of the Scriptures, 

and of his Spirit of Prophecy . . 113 
xii. Of the Efficacy of his Preaching, and of 

his Gift of Healing . . .125 

xiii. Of the Sacred Stigmata . . .137 

xiv. Of his Sufferings and Death . .148 

xv. Of his Canonisation, and the Translation 

of his Body . . . . 155 

Digitized by 

vi…
```


## 다음 단계 (결정 보류 — 측정만 보고)

본 보고서는 raw 측정치만 제시한다. 어떤 컬렉션을 재인제스트할지(임계값·범위)는 step 2 에서 사용자와 결정한다.

