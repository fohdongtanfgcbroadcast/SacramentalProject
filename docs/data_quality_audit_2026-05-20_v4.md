# 데이터 품질 감사 v4 — bunyan 재인제스트 후 (2026-05-20)

## 배경

이 보고서는 동일 일자의 v3(`docs/data_quality_audit_2026-05-20_v3.md`, 커밋 `22a6475` — vermigli EEBO-TCP 교체 후) 이후 **bunyan 컬렉션을 Step 2 strip 규칙 적용 상태로 재인제스트한 결과**를 측정한 v4 다.

v3 에서 bunyan(junk 4.0%)이 새 상위 오염 1위가 됐다. 진단 결과 bunyan은 Step 2 strip(`textclean.strip_web_chrome`, 커밋 `da11213`) 적용 *이전* 에 인제스트된 컬렉션이었고, 코드 변경 없이 **현재 strip 로직만으로 시뮬레이션 시 잡라인 0개**가 확인됐다(Gutenberg `*** START`/`*** END` 마커 슬라이싱이 이미 구현돼 있음). 즉 Step 3 에서 junk>5% 임계값으로 4개만 골랐을 때 4.0% 인 bunyan 이 제외돼 누락 보정이 필요했다.

## v3 → v4 변화 — bunyan 재인제스트 효과

| 지표 | v3 | v4 | 변화 |
|---|--:|--:|---|
| bunyan junk % | **4.0** | **0.0** | **-4.0pp** |
| bunyan severe % | 1.7 | 0.0 | -1.7pp |
| bunyan 청크 수 | 2,538 | 2,426 | -112 (strip 자연 감소) |
| 총 청크 | 599,521 | 599,409 | -112 |

**새 상위 오염 1위: confessions 3.4%** (NewAdvent CCEL + Gutenberg 혼합 컬렉션, Class A — 동일 패턴 누락 보정. 재인제스트만으로 해소 가능 예상).

> **ocr % 한글 소스 오탐 경고**: OCR 휴리스틱은 알파벳을 `[A-Za-z]` 로만 센다. `moltmann` ocr 99.5% 는 OCR 깨짐이 아니라 한글 본문 — 무시할 것.

## 측정 방법

- 전수 스캔: 58 컬렉션 / 총 599,409 청크 (샘플링 아님)
- **junk %**: `_IA_JUNK`/`_JUNK_RE` 매칭 또는 `_NAV_LABELS` 포함 라인이 1개 이상인 청크 비율 (= step 2 가 실제 제거할 대상). 측정 심볼은 `translate_confessions` 에서 직접 import — 정의 일치 보장
- **severe %**: `_drop_boilerplate_lines` 적용 시 길이가 30% 초과 줄어드는 청크 비율 (심한 오염)
- **평균 축소**: 전 청크 평균 `_drop_boilerplate_lines` 축소율
- **ocr %**(별도 축, **step 2 범위 외**): 비알파 비율 50% 초과 라인 포함 청크. strip 로직이 다루지 않음 — 보존처리/별도 작업 대상

> 주의: 청크 오버랩 200자로 경계 잡음이 두 청크에 중복 출현할 수 있다. "오염 청크 비율"로는 정직한 수치다(어느 사본이 검색돼도 잡음 노출). 중복 제거는 하지 않는다.

## 컬렉션별 오염도 (junk % 내림차순)

| 컬렉션 | 청크 | junk % | severe % | ocr % | 평균축소 % |
|---|--:|--:|--:|--:|--:|
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
| bunyan | 2,426 | 0.0 | 0.0 | 1.2 | 0.0 |
| vermigli | 21,593 | 0.0 | 0.0 | 6.5 | 0.0 |
| benedict | 246 | 0.0 | 0.0 | 17.1 | 0.0 |
| wesley | 1,921 | 0.0 | 0.0 | 0.9 | 0.0 |

## 상위 3개 오염 컬렉션 — 잡음 청크 표본 (원문 그대로)


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


### edwards (junk 3.2%)

**표본 1:**
```
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML+RDFa 1.1//EN"
 "http://www.w3.org/MarkUp/DTD/xhtml-rdfa-2.dtd">
<html xmlns= "http://www.w3.org/1999/xhtml"
 lang= "en"
 xml:lang="en"
 xmlns:dc="http://purl.org/dc/terms/">
 <head>
 <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
 <meta name='uid' value='0' />
 <meta name='uname' value='[not logged in]' />
 <meta name='umail' value='[none]' />
 <meta name='isAdmin' value='0' />
 <meta name='debug' value='0' />
 <meta name="csrf-token" content="zCDQK87KEGPAZoOs00NVTjhofec6JK8BLxAM1rXa" />
 <meta name="pageType" value='WorkInfo' />
…
```

**표본 2:**
```
rf-token" content="zCDQK87KEGPAZoOs00NVTjhofec6JK8BLxAM1rXa" />
 <meta name="pageType" value='WorkInfo' />
 
 <title>
 Work info: Religious Affections -
 Christian Classics Ethereal Library
</title>

<meta charset="UTF-8" />
<meta http-equiv="content-type"
 content="application/xhtml+xml; charset=utf-8" />

<meta name="viewport" content="width=device-width, initial-scale=1.0" />

<link rel="search" type="application/opensearchdescription+xml"
 title="CCEL" href="/ccelsearch.xml" />
```

**표본 3:**
```
; charset=utf-8" />

<meta name="viewport" content="width=device-width, initial-scale=1.0" />

<link rel="search" type="application/opensearchdescription+xml"
 title="CCEL" href="/ccelsearch.xml" />

<link rel="shortcut icon" type="image/x-icon"
 href="/img/favicon.ico" />
<link rel="apple-touch-icon"
 href="https://ccel.org/img/apple-touch-icon.png" />
<link rel="apple-touch-icon" sizes="114x114"
 href="https://ccel.org/img/apple-touch-icon@2x.png" />
<link rel="apple-touch-icon" sizes="72x72"
 href="https://ccel.org/img/apple-touch-icon_ipad.png" />
<link rel="apple-touch-icon" sizes="144x14…
```


## 다음 단계 (결정 보류 — 측정만 보고)

본 보고서는 raw 측정치만 제시한다. 어떤 컬렉션을 재인제스트할지(임계값·범위)는 step 2 에서 사용자와 결정한다.

