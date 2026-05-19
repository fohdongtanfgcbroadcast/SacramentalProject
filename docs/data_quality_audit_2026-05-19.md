# 데이터 품질 감사 — RAG 코퍼스 잡음 측정 (2026-05-19)

## 배경

`src/symposium/ingest.py` 는 `clean_text` 만 적용하고 `scripts/translate_confessions.py` 의 라인레벨 잡음 제거(`_IA_JUNK`/`_JUNK_RE`/`_NAV_LABELS`/`_drop_boilerplate_lines`)를 쓰지 않는다. 본 보고서는 그 결과 검색/인용 코퍼스에 남은 디지털화 잡음·웹 chrome·네비 라벨의 오염도를 컬렉션별로 정량화한다. **측정 전용 — 데이터 미변경.**

## 측정 방법

- 전수 스캔: 58 컬렉션 / 총 599,860 청크 (샘플링 아님)
- **junk %**: `_IA_JUNK`/`_JUNK_RE` 매칭 또는 `_NAV_LABELS` 포함 라인이 1개 이상인 청크 비율 (= step 2 가 실제 제거할 대상). 측정 심볼은 `translate_confessions` 에서 직접 import — 정의 일치 보장
- **severe %**: `_drop_boilerplate_lines` 적용 시 길이가 30% 초과 줄어드는 청크 비율 (심한 오염)
- **평균 축소**: 전 청크 평균 `_drop_boilerplate_lines` 축소율
- **ocr %**(별도 축, **step 2 범위 외**): 비알파 비율 50% 초과 라인 포함 청크. strip 로직이 다루지 않음 — 보존처리/별도 작업 대상

> 주의: 청크 오버랩 200자로 경계 잡음이 두 청크에 중복 출현할 수 있다. "오염 청크 비율"로는 정직한 수치다(어느 사본이 검색돼도 잡음 노출). 중복 제거는 하지 않는다.

> **ocr % 한글 소스 오탐 경고**: OCR 휴리스틱은 알파벳을 `[A-Za-z]` 로만 센다. 한글 소스 컬렉션은 한글 글자가 전부 비알파로 계산돼 ocr % 가 비정상적으로 부풀려진다. **`moltmann` ocr 99.5% 는 OCR 깨짐이 아니라 한글 본문(몰트만 선집, 한글 PDF 소스)** 이다 — 무시할 것. 영문 소스 컬렉션의 ocr % 만 의미가 있다(그조차 헬라어/라틴어 음역·각주기호·구절번호로 일부 과대). ocr 축은 애초에 step 2 범위 외(strip 로직 미적용 영역)이므로 step 2 판단에는 영향 없음.

## 컬렉션별 오염도 (junk % 내림차순)

| 컬렉션 | 청크 | junk % | severe % | ocr % | 평균축소 % |
|---|--:|--:|--:|--:|--:|
| murray | 1,095 | 9.3 | 3.7 | 6.9 | 2.5 |
| wesley | 2,126 | 7.5 | 3.8 | 1.5 | 2.4 |
| zinzendorf | 1,202 | 6.4 | 0.0 | 10.1 | 0.3 |
| watts | 1,285 | 5.2 | 1.9 | 11.5 | 1.4 |
| vermigli | 21,507 | 4.2 | 0.0 | 2.5 | 0.3 |
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
| jerome | 5,234 | 0.0 | 0.0 | 2.8 | 0.0 |
| benedict | 246 | 0.0 | 0.0 | 17.1 | 0.0 |

## 상위 3개 오염 컬렉션 — 잡음 청크 표본 (원문 그대로)


### murray (junk 9.3%)

**표본 1:**
```
﻿The Project Gutenberg eBook of Lord, Teach Us To Pray
 
This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this eBook or online
at www.gutenberg.org. If you are not located in the United States,
you will have to check the laws of the country where you are located
before using this eBook.

Title: Lord, Teach Us To Pray

Author: Andrew Murray

 
Release date: September 27, 2008 [eBook #26…
```

**표본 2:**
```
ay

Author: Andrew Murray

 
Release date: September 27, 2008 [eBook #26709]
 Most recently updated: January 4, 2021

Language: English

Other information and formats: www.gutenberg.org/ebooks/26709

Credits: Produced by Free Elf, Jeannie Howse and the Online
 Distributed Proofreading Team at https://www.pgdp.net (This
 file was produced from images generously made available
 by The Internet Archive)

Produced by Free Elf, Jeannie Howse and the Online
Distributed Proofreading Team at https://www.pgdp.net (This
file was produced from images generously made available
by The Internet Archive)

 *…
```

**표본 3:**
```
ced by Free Elf, Jeannie Howse and the Online
Distributed Proofreading Team at https://www.pgdp.net (This
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

Copyright, 1896, by HENRY ALTEMUS…
```


### wesley (junk 7.5%)

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
 <meta name="csrf-token" content="eHoKUpv7GP5qYSHXpe5In791L4dtjBPKnHwM9EFZ" />
 <meta name="pageType" value='WorkInfo' />
…
```

**표본 2:**
```
" content="eHoKUpv7GP5qYSHXpe5In791L4dtjBPKnHwM9EFZ" />
 <meta name="pageType" value='WorkInfo' />
 
 <title>
 Work info: Sermons on Several Occasions -
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


### zinzendorf (junk 6.4%)

**표본 1:**
```
r People, confor- 
mably to Article che twentieth, inceſſantly 
give the following Explanation, 1.) That our 
Wozks are not able fo reconcile us with God 
and purchaſe Grace, but this is effected only 
throngh Faith; ) that our Sins are fo2- 

given us fo: CYRIST'S Sake, Whoſo- ® 

ever now ſuppoſes, that he can by Moꝛks 

accompliſh this, and merit Grace, he deſpifes | 

; Chrift, the only Pedialoz between God and 

Pen, 

ba, hos 

oi % ww ww Yy yt” SF T3P ws 

© LING: 
$03.4 

A Synodal Writing, &c. xvii 
Men, and his Propitiatory Sacrifice, and ſeeks a 
Way of his own toGod, contrary te th…
```

**표본 2:**
```
od and 

Pen, 

ba, hos 

oi % ww ww Yy yt” SF T3P ws 

© LING: 
$03.4 

A Synodal Writing, &c. xvii 
Men, and his Propitiatory Sacrifice, and ſeeks a 
Way of his own toGod, contrary te the Goſpel, 

The Auf ſburg Contetiion appcals, in this 
Doctrine of Faith, to the expreſs and clear 
Words of Paul in ions Places, particularly 
in Epheſ. ii. By Grace ye are ſaved, through 
Faith, and that not of yourſelves, but it is the 
Gift of God; not of Works, leſt any Man fhould 
boaſt: And proves out of St. Auguſtine, (who 

treats of this Point diligently, and reaches 

the ſame) that we through Fait…
```

**표본 3:**
```
allo remark, 

that we here ſpeak 10 of uch a Faith, wher 

a Man coins to himſelf a Thought crying out, 

e believe; nor of an Aﬀent which any one 

gives to the Hiſtory of Chrilt's having ſuffer- 

ed and riſen again from the dead, either out 
of an indclent Credulity, after the Manner of 
natural Men, or out of an unhappy "AvToie, 
like Satan: but we ſpeak of true Faith in 
Chriſt, which believes that we through Vim 
do obtain Grace and Fo2giveneſs of Sins. 
D0 that a wrt knows * through * 

Re 

* 

© wwwilhninw w \\ ww tl Jy eee 

>» 

A Synedal Writing, &c. XIX 

he has a gracious God, a…
```


## 다음 단계 (결정 보류 — 측정만 보고)

본 보고서는 raw 측정치만 제시한다. 어떤 컬렉션을 재인제스트할지(임계값·범위)는 step 2 에서 사용자와 결정한다.

