# 데이터 품질 감사 — RAG 코퍼스 잡음 측정 (2026-07-14)

## 배경

`src/symposium/ingest.py` 는 `clean_text` 만 적용하고 `scripts/translate_confessions.py` 의 라인레벨 잡음 제거(`_IA_JUNK`/`_JUNK_RE`/`_NAV_LABELS`/`_drop_boilerplate_lines`)를 쓰지 않는다. 본 보고서는 그 결과 검색/인용 코퍼스에 남은 디지털화 잡음·웹 chrome·네비 라벨의 오염도를 컬렉션별로 정량화한다. **측정 전용 — 데이터 미변경.**

## 측정 방법

- 전수 스캔: 58 컬렉션 / 총 567,448 청크 (샘플링 아님)
- **junk %**: `_IA_JUNK`/`_JUNK_RE` 매칭 또는 `_NAV_LABELS` 포함 라인이 1개 이상인 청크 비율 (= step 2 가 실제 제거할 대상). 측정 심볼은 `translate_confessions` 에서 직접 import — 정의 일치 보장
- **severe %**: `_drop_boilerplate_lines` 적용 시 길이가 30% 초과 줄어드는 청크 비율 (심한 오염)
- **평균 축소**: 전 청크 평균 `_drop_boilerplate_lines` 축소율
- **ocr %**(별도 축, **step 2 범위 외**): 비알파 비율 50% 초과 라인 포함 청크. strip 로직이 다루지 않음 — 보존처리/별도 작업 대상

> 주의: 청크 오버랩 200자로 경계 잡음이 두 청크에 중복 출현할 수 있다. "오염 청크 비율"로는 정직한 수치다(어느 사본이 검색돼도 잡음 노출). 중복 제거는 하지 않는다.

## 컬렉션별 오염도 (junk % 내림차순)

| 컬렉션 | 청크 | junk % | severe % | ocr % | 평균축소 % |
|---|--:|--:|--:|--:|--:|
| confessions | 4,776 | 0.3 | 0.0 | 2.5 | 0.0 |
| murray | 963 | 0.2 | 0.0 | 7.6 | 0.0 |
| cyril_jerusalem | 1,719 | 0.2 | 0.0 | 5.5 | 0.0 |
| moltmann | 12,114 | 0.2 | 0.0 | 2.6 | 0.0 |
| kuyper | 4,126 | 0.1 | 0.0 | 1.0 | 0.0 |
| schweitzer | 3,193 | 0.1 | 0.0 | 4.9 | 0.0 |
| harnack | 5,530 | 0.1 | 0.0 | 1.9 | 0.0 |
| clement_alexandria | 4,516 | 0.0 | 0.0 | 10.6 | 0.0 |
| schleiermacher | 9,051 | 0.0 | 0.0 | 2.8 | 0.0 |
| cyprian | 4,612 | 0.0 | 0.0 | 9.9 | 0.0 |
| irenaeus | 4,745 | 0.0 | 0.0 | 8.5 | 0.0 |
| rauschenbusch | 2,656 | 0.0 | 0.0 | 2.5 | 0.0 |
| knox | 8,016 | 0.0 | 0.0 | 5.3 | 0.0 |
| luther | 14,195 | 0.0 | 0.0 | 2.6 | 0.0 |
| baxter | 18,426 | 0.0 | 0.0 | 2.7 | 0.0 |
| cranmer | 12,023 | 0.0 | 0.0 | 5.0 | 0.0 |
| basil | 8,050 | 0.0 | 0.0 | 4.4 | 0.0 |
| zwingli | 4,123 | 0.0 | 0.0 | 0.7 | 0.0 |
| hodge | 13,459 | 0.0 | 0.0 | 2.9 | 0.0 |
| mather | 14,313 | 0.0 | 0.0 | 1.4 | 0.0 |
| gregory_i | 15,554 | 0.0 | 0.0 | 14.2 | 0.0 |
| ambrose | 5,185 | 0.0 | 0.0 | 8.3 | 0.0 |
| origen | 5,194 | 0.0 | 0.0 | 6.8 | 0.0 |
| john_damascus | 6,253 | 0.0 | 0.0 | 7.7 | 0.0 |
| bullinger | 10,528 | 0.0 | 0.0 | 9.1 | 0.0 |
| wesley | 11,967 | 0.0 | 0.0 | 8.5 | 0.0 |
| athanasius | 25,261 | 0.0 | 0.0 | 7.3 | 0.0 |
| calvin | 120,079 | 0.0 | 0.0 | 14.3 | 0.0 |
| aquinas | 39,283 | 0.0 | 0.0 | 19.9 | 0.0 |
| justin_martyr | 1,977 | 0.0 | 0.0 | 3.1 | 0.0 |
| owen | 2,725 | 0.0 | 0.0 | 5.4 | 0.0 |
| whitefield | 9,762 | 0.0 | 0.0 | 9.4 | 0.0 |
| julian_norwich | 712 | 0.0 | 0.0 | 28.9 | 0.0 |
| rutherford | 3,813 | 0.0 | 0.0 | 5.3 | 0.0 |
| ritschl | 3,249 | 0.0 | 0.0 | 1.8 | 0.0 |
| watts | 1,215 | 0.0 | 0.0 | 11.5 | 0.0 |
| eckhart | 708 | 0.0 | 0.0 | 10.0 | 0.0 |
| anselm | 2,392 | 0.0 | 0.0 | 20.5 | 0.0 |
| celano | 1,317 | 0.0 | 0.0 | 4.9 | 0.0 |
| francis | 2,971 | 0.0 | 0.0 | 1.7 | 0.0 |
| zinzendorf | 1,184 | 0.0 | 0.0 | 10.2 | 0.0 |
| kierkegaard | 1,006 | 0.0 | 0.0 | 0.9 | 0.0 |
| jerome | 5,234 | 0.0 | 0.0 | 2.8 | 0.0 |
| edwards | 3,919 | 0.0 | 0.0 | 1.8 | 0.0 |
| forsyth | 2,388 | 0.0 | 0.0 | 0.0 | 0.0 |
| bavinck | 1,961 | 0.0 | 0.0 | 3.9 | 0.0 |
| bunyan | 2,426 | 0.0 | 0.0 | 1.2 | 0.0 |
| chrysostom | 33,970 | 0.0 | 0.0 | 8.1 | 0.0 |
| scotus | 1,379 | 0.0 | 0.0 | 7.1 | 0.0 |
| law | 3,754 | 0.0 | 0.0 | 2.5 | 0.0 |
| vermigli | 21,593 | 0.0 | 0.0 | 5.9 | 0.0 |
| melanchthon | 1,503 | 0.0 | 0.0 | 4.4 | 0.0 |
| benedict | 246 | 0.0 | 0.0 | 17.1 | 0.0 |
| tertullian | 7,195 | 0.0 | 0.0 | 10.3 | 0.0 |
| kempis | 701 | 0.0 | 0.0 | 28.4 | 0.0 |
| augustine | 50,832 | 0.0 | 0.0 | 20.8 | 0.0 |
| bernard | 2,254 | 0.0 | 0.0 | 18.0 | 0.0 |
| bonaventure | 5,152 | 0.0 | 0.0 | 5.7 | 0.0 |

## 상위 3개 오염 컬렉션 — 잡음 청크 표본 (원문 그대로)


### confessions (junk 0.3%)

**표본 1:**
```
-NRLF 

Digitized by the Internet Archive 

in 2007 with funding from 

IVIicrosoft Corporation 

THE 

CATECHISM 

OP THE 

CHURCH OF GENEVA, 

BY THE REV. JOH^ CJALVIN. 
II 

TRAirSULTEI) FEOM THE lATHTy'^ 

BY THE REV. ELUAH WATEU^iAN\ 

Author of the Life itf Calvin. r\\ 

AN APPENDIX, 

IN A LETTER ASSBESSEJ) TO 
WILLIAM S. JOHNSON, L. L. D. 

Showing that ** the Catechism commonly called Dr. Alexande" 

Noweir's,** which was sanctioned in the Convocation of Bishopa 

and Clergy in 1562, and publLslied 1570, " as a standing 

summary of tiie doctrines of the English Church,** is 

in subs…
```

**표본 2:**
```
English Church,** is 

in substance the Catecliism of cSvin enlarged. 

(etS ri]¥ KctTTj^TtTt))^ fAotidein Kxt Tfi^it tiq ray ctitouft^ 
Ex Prefa, CyriiU Catechaeon p, 8. ejiit Operum^ 

HAETFORD : 

Sheldon ^ Good'u.in.....JPrmtfr.'}y 

2>ti0!ttwt of i3Dottttecticut, 0^r 

yff^t^f^^ff BE IT REMEMBERED : That on the fifth day of 
sT7s$ August, in the thirty-niiith year of the Independence 
'IZ^ of the United States of America, Eliiah Waterman, 
^^^^^^ of the said District, hath dei>osited in "this office, the 
title of a Book, the riglit whereof he claims as author, in the 
words following to …
```

**표본 3:**
```
nd dulled the sense of heading, have 
still left to you, the vigour of your under- 
standing, the warm devotior^ of your heart, 
and the eloc|uenee of your tongue, to vindi- 
cate in yaiir social cirde, the purity of the 
scriptures, the unity of the Church, and the 
godhead of, the Redeemer. With due res- 
pect for your learning and piety, and ac- 
knowledgment of the favours I have receiv- 
ed from you in the free use of your valuable 
library ; I sincerely pray God to continue 
your health and social comforts, and to pro- 
long your days to see the prosperity of Zion ; 
and that he would su…
```


### murray (junk 0.2%)

**표본 1:**
```
Produced by Free Elf, Jeannie Howse and the Online
file was produced from images generously made available
by The Internet Archive)

 * * * * *

 +-----------------------------------------------------------+
 | Transcriber's Note: |
 | |
 | Obvious typographical errors have been corrected. For |
 | a complete list, please see the end of this document. |
 | |
 +-----------------------------------------------------------+

 * * * * *

Lord, Teach Us
To Pray

By Rev. Andrew Murray

Philadelphia
Henry Altemus

Copyright, 1896, by HENRY ALTEMUS.

LORD, TEACH US TO PRAY

OR

THE ONLY TEACHER.
```

**표본 2:**
```
Produced by Heiko Evermann, Nigel Blower and the Online
file was produced from images generously made available
by The Internet Archive/Canadian Libraries)

 THE
 MINISTRY OF INTERCESSION

 A PLEA FOR MORE PRAYER

 BY THE

 REV. ANDREW MURRAY

 WELLINGTON, S. AFRICA

 AUTHOR OF
 "THE HOLIEST OF ALL" "ABIDE IN CHRIST"
 "WAITING ON GOD" "THE LORD'S TABLE"
 ETC. ETC.

 "I have set watchmen upon thy walls, O Jerusalem,
 which shall never hold their peace day nor night:
 ye that are the Lord's remembrancers, keep not
 silence, and give Him no rest, till He establish,
 and till He make Jerusalem a p…
```


### cyril_jerusalem (junk 0.2%)

**표본 1:**
```
•NRLF 

\j r- 

LIBRARY OF FATHERS 

OF THE 

HOLY CATHOLIC CHURCH, 

ANTERIOR TO THE DIVISION OF THE EAST AND WEST. 

TO THE BINDER. 

The Binder is desired to Cancel the General Titles to the 
first eight volumes, and to place these in their stead. 

\ 

YET SHALL NOT THY TEACHERS BE REMOVED INTO A CORNER ANY MORE, B(7T 
THINE EYES SHALL SEE THY TEACHERS. Isaiah xxx. 20. 

OXFORD, 

JOHN HENRY PARKER; 

J. G. F. AND J. RIVINGTON, LONDON. 

MDCCCXLII. 

LIBRARY OF FATHERS 

OF THE 

HOLY CATHOLIC CHURCH, 

ANTERIOR TO THE DIVISION OF THE EAST AND WEST. 

TRANSLATED B\ MEMBERS OF THE ENGLISH C…
```

**표본 2:**
```
TED B\ MEMBERS OF THE ENGLISH CHURCH. 

V. -2. 

YET SHALL NOT THY TEACHERS BE REMOVED INTO A CORNER ANY MORE, BUT 
THINE EYES SHALL SEE THY TEACHERS. Isaiah xxx. 20. 

OXFORD, 

JOHN HENRY PARKER; 

J, G. F. AND J. RIVINGTON, LONDON, 

MDCCCXLII. 

Digitized by the Internet Archive 

in 2008 with funding from 

IVIicrosoft Corporation 

Us: 

TO THB 
MOST BEVEREND FATHER IN GOD 

WILLIAM 

LORD ARCHBISHOP OF CANTERBURY, 
PRIMATE OF ALL ENGLAND, 

FORMERLY REGIUS PROFESSOR OF DIVINITY IN THE UNIVERSITY OF OXFORD, 

THIS LIBRARY 

OF 

ANCIENT BISHOPS, FATHERS, DOCTORS, MARTYRS, CONFESSORS, 
OF…
```

**표본 3:**
```
ield Station 
University of California 
Richmond, CA 94804-4698 

ALL BOOKS MAY BE RECALLED AFTER 7 DAYS 
2-month loans may be renewed by calling 

(415) 642-6753 
1-year loans may be recharged by bringing books 

to NRLF 
Renewals and recharges may be made 4 days 

prior to due date 

DUE AS STAMPED BELOW 

JIIN 3 0 199? 

sE^^toNILL 

J^N^5?oo2 

U, C- BFRKELEY 

S2f926 

THE UNIVERSITY OF CAUFORNIA UBRARY
```


## 다음 단계 (결정 보류 — 측정만 보고)

본 보고서는 raw 측정치만 제시한다. 어떤 컬렉션을 재인제스트할지(임계값·범위)는 step 2 에서 사용자와 결정한다.

