# 데이터 품질 감사 v5 — wesley 정본 교체 후 (2026-05-20)

## 배경

이 보고서는 동일 일자의 v4(`docs/data_quality_audit_2026-05-20_v4.md`, 커밋 `526d5f7` — bunyan 재인제스트 후) 이후 **wesley 컬렉션의 stub 책 3종을 CCEL 정본으로 교체하고 재인제스트한 결과**를 측정한 v5 다.

v2 audit 분석에서 wesley 가 Class B(HTML stub 데이터 결손) 로 분류돼 있었다: `sermons.txt` / `christian_perfection.txt` / `journal_vol1.txt` 가 잘못 받은 CCEL Work Info HTML 메타 페이지(각 28KB) 였고, 실 본문이 stub 으로 소실된 상태. junk % 는 0% 였지만 *데이터 자체가 없는 것* 이 문제였다.

CCEL 의 cache plain text 엔드포인트(`https://ccel.org/ccel/w/wesley/<work>/cache/<work>.txt`) 에서 정본 3종을 받아 동일 파일명으로 교체했다(기존 stub 은 `*.ccel_stub_legacy.html` 로 백업). 모두 Public Domain.

## v4 → v5 변화 — wesley 정본 교체 효과

| 지표 | v4 (stub) | v5 (CCEL 정본) | 변화 |
|---|--:|--:|---|
| wesley 청크 수 | 1,921 | **12,407** | **+10,486 (6.5배)** |
| wesley junk % | 0.0 | 0.0 | — |
| wesley ocr % | 0.9 | 8.7 | +7.8pp* |
| 총 청크 | 599,409 | 609,895 | +10,486 |

\* ocr % 증가는 손상이 아니라 CCEL 정본의 그리스어/라틴어 인용·각주·메타 헤더(`Title:`/`Creator(s):` 등 비알파 라인) 포함에 따른 측정 휴리스틱 산물. junk % 와 무관하며 검색 품질에 무해.

**파일별 청크 분포:**
- `sermons.txt` (4.0MB, CCEL): 신규 (이전 stub 2~3 청크 → 본문 회복)
- `christian_perfection.txt` (0.2MB, CCEL): 신규
- `journal_vol1.txt` (0.94MB, CCEL, 사실상 일지 전체 합본): 신규
- `sermons_vol2.txt` (0.5MB, 기존): 998 청크 유지
- `earnest_appeal.txt` (0.5MB, 기존): 915 청크 유지

**잔여 상위 오염 (v5):**
1. confessions 3.4% (NewAdvent CCEL + Gutenberg 혼합, Class A — 재인제스트로 해소 가능)
2. kierkegaard 3.3%, edwards 3.2%, luther 2.6% (이하 모두 Class A, 동일 처리)

→ Step 5 의 wesley B 완료. 잔여 zinzendorf/watts(Class C OCR) 만 별도 결정 대기.

> **ocr % 한글 소스 오탐 경고**: OCR 휴리스틱은 알파벳을 `[A-Za-z]` 로만 센다. `moltmann` ocr 99.5% 는 OCR 깨짐이 아니라 한글 본문 — 무시할 것.

## 측정 방법

- 전수 스캔: 58 컬렉션 / 총 609,895 청크 (샘플링 아님)
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
| wesley | 12,407 | 0.0 | 0.0 | 8.7 | 0.0 |
| justin_martyr | 1,977 | 0.0 | 0.0 | 3.1 | 0.0 |
| watts | 1,215 | 0.0 | 0.0 | 11.5 | 0.0 |
| zinzendorf | 1,184 | 0.0 | 0.0 | 10.2 | 0.0 |
| jerome | 5,234 | 0.0 | 0.0 | 2.8 | 0.0 |
| bunyan | 2,426 | 0.0 | 0.0 | 1.2 | 0.0 |
| vermigli | 21,593 | 0.0 | 0.0 | 6.5 | 0.0 |
| benedict | 246 | 0.0 | 0.0 | 17.1 | 0.0 |

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

