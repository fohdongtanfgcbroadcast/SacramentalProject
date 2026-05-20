# 데이터 품질 감사 v6 — Class A 7개 일괄 재인제스트 후 (2026-05-21)

## 배경

이 보고서는 전일(2026-05-20) v5(커밋 `a5496fc`, wesley 정본 회복 후) 이후 **Class A 잔여 7개 컬렉션을 Step 2 strip 적용 상태로 일괄 재인제스트한 결과**를 측정한 v6 다.

v5 잔여 상위 오염은 모두 bunyan 과 동일 패턴(Step 2 strip 적용 *이전* 인제스트가 남은 상태) 으로 진단됐다. 코드 변경 없이 `textclean.strip_web_chrome` 만으로 시뮬레이션 시 5/7 은 잡라인 ≤1개, 2개(confessions 13개, harnack 25개) 만 잔여 — 청크 단위로는 모두 1% 이하 예상.

## v5 → v6 변화 — Class A 일괄 재인제스트 효과

| 컬렉션 | v5 청크 | v6 청크 | v5 junk % | v6 junk % | Δ |
|---|--:|--:|--:|--:|--:|
| confessions | 6,470 | 4,776 | 3.4 | **0.3** | **-3.1pp** |
| kierkegaard | 1,040 | 1,006 | 3.3 | **0.0** | **-3.3pp** |
| edwards | 4,032 | 3,919 | 3.2 | **0.0** | **-3.2pp** |
| luther | 14,941 | 14,433 | 2.6 | **0.0** | **-2.6pp** |
| harnack | 5,841 | 5,530 | 2.1 | **0.1** | **-2.0pp** |
| melanchthon | 1,541 | 1,503 | 2.1 | **0.0** | **-2.1pp** |
| schweitzer | 3,253 | 3,193 | 2.1 | **0.1** | **-2.0pp** |

총 청크 609,895 → 607,137 (-2,758, strip 자연 감소). confessions 가 -26% 로 가장 큰 감소(NewAdvent CCEL boilerplate 다수 제거).

**새 상위 오염 1위: whitefield 1.7%** — 동일 패턴(Step 2 누락 보정 후보).

## 코퍼스 전체 상태

이 시점에서 **junk % > 2% 컬렉션이 없다.** 가장 높은 whitefield 1.7%, anselm 1.4%, francis 1.3% 등 — 모두 추가 일괄 재인제스트로 해소 가능 범위.

남은 *구조적* 잡음은 OCR Class C(strip 무효):
- zinzendorf · watts(OCR 손상), moltmann ocr 99.5%(한글 본문 오탐) 정도.

> **ocr % 한글 소스 오탐 경고**: OCR 휴리스틱은 알파벳을 `[A-Za-z]` 로만 센다. `moltmann` ocr 99.5% 는 OCR 깨짐이 아니라 한글 본문 — 무시할 것.

## 측정 방법

- 전수 스캔: 58 컬렉션 / 총 607,137 청크 (샘플링 아님)
- **junk %**: `_IA_JUNK`/`_JUNK_RE` 매칭 또는 `_NAV_LABELS` 포함 라인이 1개 이상인 청크 비율 (= step 2 가 실제 제거할 대상). 측정 심볼은 `translate_confessions` 에서 직접 import — 정의 일치 보장
- **severe %**: `_drop_boilerplate_lines` 적용 시 길이가 30% 초과 줄어드는 청크 비율 (심한 오염)
- **평균 축소**: 전 청크 평균 `_drop_boilerplate_lines` 축소율
- **ocr %**(별도 축, **step 2 범위 외**): 비알파 비율 50% 초과 라인 포함 청크. strip 로직이 다루지 않음 — 보존처리/별도 작업 대상

> 주의: 청크 오버랩 200자로 경계 잡음이 두 청크에 중복 출현할 수 있다. "오염 청크 비율"로는 정직한 수치다(어느 사본이 검색돼도 잡음 노출). 중복 제거는 하지 않는다.

## 컬렉션별 오염도 (junk % 내림차순)

| 컬렉션 | 청크 | junk % | severe % | ocr % | 평균축소 % |
|---|--:|--:|--:|--:|--:|
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
| irenaeus | 4,745 | 0.0 | 0.0 | 8.5 | 0.0 |
| luther | 14,433 | 0.0 | 0.0 | 2.9 | 0.0 |
| cranmer | 12,023 | 0.0 | 0.0 | 5.0 | 0.0 |
| wesley | 12,407 | 0.0 | 0.0 | 8.7 | 0.0 |
| justin_martyr | 1,977 | 0.0 | 0.0 | 3.1 | 0.0 |
| watts | 1,215 | 0.0 | 0.0 | 11.5 | 0.0 |
| zinzendorf | 1,184 | 0.0 | 0.0 | 10.2 | 0.0 |
| kierkegaard | 1,006 | 0.0 | 0.0 | 0.9 | 0.0 |
| jerome | 5,234 | 0.0 | 0.0 | 2.8 | 0.0 |
| edwards | 3,919 | 0.0 | 0.0 | 1.8 | 0.0 |
| bunyan | 2,426 | 0.0 | 0.0 | 1.2 | 0.0 |
| vermigli | 21,593 | 0.0 | 0.0 | 6.5 | 0.0 |
| melanchthon | 1,503 | 0.0 | 0.0 | 4.4 | 0.0 |
| benedict | 246 | 0.0 | 0.0 | 17.1 | 0.0 |

## 상위 3개 오염 컬렉션 — 잡음 청크 표본 (원문 그대로)


### whitefield (junk 1.7%)

**표본 1:**
```
﻿The Project Gutenberg eBook of The works of the Reverend George Whitefield, M.A., Vol. 1 (of 6)
 
This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this eBook or online
at www.gutenberg.org. If you are not located in the United States,
you will have to check the laws of the country where you are located
before using this eBook.

Title: The works of the Reverend George Whitefield, M.A.,…
```

**표본 2:**
```
s of the Reverend George Whitefield, M.A., Vol. 1 (of 6)
 Containing all his sermons and tracts, etc.

Author: George Whitefield

 
Release date: September 12, 2022 [eBook #68976]

Language: English

Original publication: United Kingdom: Printed for Edward and Charles Dilly, in the Poultry; and Messers. Kincaid and Bell, at Edinburgh, 1771

Other information and formats: www.gutenberg.org/ebooks/68976

Credits: Brian Wilson, Heather Clark and the Online Distributed Proofreading Team at https://www.pgdp.net (This file was produced from images generously made available by The Internet Archive)

…
```

**표본 3:**
```
nline Distributed Proofreading Team at https://www.pgdp.net (This file was produced from images generously made available by The Internet Archive)

 The Works of the Reverend George Whitefield, M.A.

 ┌────────────────────────────────────────────────────────────────┐
 │ │
 │ Transcriber’s Notes │
 │ │
 │ │
 │ Punctuation has been standardized. │
 │ │
 │ The original text may show quotations within quotations, set │
 │ off by similar quote marks. The inner quotations have been │
 │ changed to alternate quote marks for improved readability. │
 │ │
 │ Characters in small caps have been replaced b…
```


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


### anselm (junk 1.4%)

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
 <meta name="csrf-token" content="o1mlCws5oYfEovz3ittgLtKvDb6sgDfjnqQd4TJF" />
 <meta name="pageType" value='WorkInfo' />
…
```

**표본 2:**
```
ta name="pageType" value='WorkInfo' />
 
 <title>
 Work info: Proslogium; Monologium; An Appendix in Behalf of the Fool by Gaunilon; and Cur Deus Homo -
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

