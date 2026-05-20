# 데이터 품질 감사 — RAG 코퍼스 잡음 측정 (2026-05-20)

## 배경

`src/symposium/ingest.py` 는 `clean_text` 만 적용하고 `scripts/translate_confessions.py` 의 라인레벨 잡음 제거(`_IA_JUNK`/`_JUNK_RE`/`_NAV_LABELS`/`_drop_boilerplate_lines`)를 쓰지 않는다. 본 보고서는 그 결과 검색/인용 코퍼스에 남은 디지털화 잡음·웹 chrome·네비 라벨의 오염도를 컬렉션별로 정량화한다.

본 v2 는 **Step 2(ingest에 strip 단일 출처 통합, 커밋 `da11213`) + Step 3(junk>5% 4개 컬렉션 재인제스트) 효과를 검증하기 위한 재측정**이다. v1(2026-05-19, 커밋 `a21b459`) 과 동일 스크립트·동일 잡음 정의로 측정했다.

## v1 대비 변화 — Step 2/3 효과 검증

junk>5% 였던 4개 컬렉션(사용자 선정 범위) 모두 임계값 아래로 진입:

| 컬렉션 | v1 청크 | v1 junk % | v2 청크 | v2 junk % | Δjunk pp |
|---|--:|--:|--:|--:|--:|
| murray | 1,095 | 9.3 | 963 | **0.2** | **-9.1** |
| wesley | 2,126 | 7.5 | 1,921 | **0.0** | **-7.5** |
| zinzendorf | 1,202 | 6.4 | 1,184 | **0.0** | **-6.4** |
| watts | 1,285 | 5.2 | 1,215 | **0.0** | **-5.2** |

총 청크 599,860 → 599,435 (-425). 재인제스트한 4개 컬렉션의 청크 수가
strip 으로 줄어든 자연 감소다.

**잔여 과제 (이번 범위 밖):**
- **vermigli 4.2%** 가 새 상위 1위 (severe 0.0%, OCR long-s 디지털화 잡음
  — strip 무효 Class C)
- wesley = **Class B**: `sermons.txt`/`christian_perfection.txt`/
  `journal_vol1.txt` 가 잘못 받은 HTML stub. junk 청크는 사라졌지만 실
  본문이 stub 으로 소실된 책이 남는다 → Gutenberg에서 정본 재취득 필요
- zinzendorf · watts = **Class C** (OCR 손상): 청크 수준 잡음은 0% 가
  됐지만 long-s/단어깨짐은 strip 범위 밖. 더 나은 스캔 소스 필요/수용
  결정

## 측정 방법

## 측정 방법

- 전수 스캔: 58 컬렉션 / 총 599,435 청크 (샘플링 아님)
- **junk %**: `_IA_JUNK`/`_JUNK_RE` 매칭 또는 `_NAV_LABELS` 포함 라인이 1개 이상인 청크 비율 (= step 2 가 실제 제거할 대상). 측정 심볼은 `translate_confessions` 에서 직접 import — 정의 일치 보장
- **severe %**: `_drop_boilerplate_lines` 적용 시 길이가 30% 초과 줄어드는 청크 비율 (심한 오염)
- **평균 축소**: 전 청크 평균 `_drop_boilerplate_lines` 축소율
- **ocr %**(별도 축, **step 2 범위 외**): 비알파 비율 50% 초과 라인 포함 청크. strip 로직이 다루지 않음 — 보존처리/별도 작업 대상

> 주의: 청크 오버랩 200자로 경계 잡음이 두 청크에 중복 출현할 수 있다. "오염 청크 비율"로는 정직한 수치다(어느 사본이 검색돼도 잡음 노출). 중복 제거는 하지 않는다.

> **ocr % 한글 소스 오탐 경고**: OCR 휴리스틱은 알파벳을 `[A-Za-z]` 로만 센다. 한글 소스 컬렉션은 한글 글자가 전부 비알파로 계산돼 ocr % 가 비정상적으로 부풀려진다. **`moltmann` ocr 99.5% 는 OCR 깨짐이 아니라 한글 본문(몰트만 선집, 한글 PDF 소스)** 이다 — 무시할 것. 영문 소스 컬렉션의 ocr % 만 의미가 있다(그조차 헬라어/라틴어 음역·각주기호·구절번호로 일부 과대). ocr 축은 애초에 step 2 범위 외(strip 로직 미적용 영역)이므로 step 2 판단에는 영향 없음.

## 컬렉션별 오염도 (junk % 내림차순)

| 컬렉션 | 청크 | junk % | severe % | ocr % | 평균축소 % |
|---|--:|--:|--:|--:|--:|
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
| benedict | 246 | 0.0 | 0.0 | 17.1 | 0.0 |
| wesley | 1,921 | 0.0 | 0.0 | 0.9 | 0.0 |

## 상위 3개 오염 컬렉션 — 잡음 청크 표본 (원문 그대로)


### vermigli (junk 4.2%)

**표본 1:**
```
oer of Sobbtooen bnoet ft, tnfitcf) if bp 
(torching,act,anonaturatlknotolcogc, itba 
bfteonctco, toil! tetteale ©oobnto is. 

; tatty 

Godby his creatures. Part.l. of Peter Martyr. Cap.i. Pag-13- 

r tBatytbisfaicngof tye apoffle feme to 
cncttr- nibigeb! otyerplaxcsof tycfcrtptnce, toberer 
i.errlr.M. m is taken from tyc toickeb.tyc knototeoge of 
©ob. JSJatcaDln tye pfalmes t Thcfolifh 
Pfaliru4,t. man faid in his hart: There is no god.&nD again 
tt IS to.littcn: In the earth there is none thatvn- 
derttandeeh,or teeketh after God. SlnO to make 
notongrccitall, itisikio intyeBcttcbaptccof …
```

**표본 2:**
```
cate anb p?ouf= 
ocnce, fo tbatbeatertben to buna feliettfe alto* 
gityeefote. atfotobcntyepfate,tyattycrcisa 
©ob, buttyatbiebaty no tcgacotomansom* 
ings.pumtyetb not,no: bcarcty futbss call bp< 
on bun, ano fucb like, it is gatycteo tyetebp, 
tyat tyis toastyctcopmfoii , tyattycugeantcD 
tyeee is a©ob m name onlie. 3 nD tyctfotetye 
&cnpturebmlctytbattycpkncto©oo. jfo? 
tbctnit©onisnot,astycpfaincbbfm to bat 
ano as touching tycmfctucs, to be bolpen,o.i 
bauc tyc fcnitionof ©00s brtptyc bias enenas 
ifijctuerc no©oo, foefonmeb as tyep ncttbcc 
calico bponbim, norlralirDfo: hope 0: atoof 
Su…
```

**표본 3:**
```
flocrca tobat Da n -t> 
©00 bao Don bnto Daniel,mo notablic confetti and 11 
bimtobetyegteat©oo: anObptye(tp:ocIar 
mattonsbnoeramoft grettous penalfte, fop 
bao that antetyoulo btaftyemcoj fpcakeenfU 
of bis name, anb Iulianus tyeapottata,altyogb 
otyettoffcmotttotcketypettoasconlfrainebat 
bfsbeatytoacknotoleOgetyepotoeeof Cb;fft, 
fnfateng : O thou Galilean, thou haft gotten 

thevidtoric. 2nbtyebetieb(ue!s toerebtitten 
totycfamceonfeBion, ftyentyep tefflfieo ano 
etteo out, tyatjefus ©b?IH Is tyc romteanO Mitthipj; 
tyc belie oiif of ©00; anoacknotoleogeo tyat 
pc came to Defteoie tycm bc…
```


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


## 다음 단계 (결정 보류 — 측정만 보고)

본 보고서는 raw 측정치만 제시한다. 어떤 컬렉션을 재인제스트할지(임계값·범위)는 step 2 에서 사용자와 결정한다.

