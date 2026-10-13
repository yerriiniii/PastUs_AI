# PastUs AI

### 2026 GDGoC 대학 연합 프로젝트 **TripleS 5팀**의 **PastUs AI 서버**입니다.
<br>

> PastUs는 대학생들이 과거의 고민과 선택, 결과와 회고를 공유하는 커뮤니티입니다.
<br>AI 서버는 게시글의 맥락을 분석해 **대학 생활 관련 태그를 자동으로 분류**하고, 사용자의 검색어와 **의미가 유사한 경험 사례를 추천**합니다.

<br><br><br><br>

## ✨ 주요 기능

| 기능 | 설명 |
| --- | --- |
| 자동 태깅 | ELECTRA 기반 다중 라벨 분류 및 태그별 신뢰도 반환 |
| 임베딩 인덱싱 | 상황·행동별 임베딩 생성 및 저장·갱신 |
| 태그 인덱싱 | 인덱싱된 게시글의 태그 ID·이름 저장 및 갱신 |
| 유사 사례 추천 | 상황·결정의 의미 유사도 기반 후보 조회 및 태그 유사도 기반 재정렬 |

<br><br><br><br>

## 🛠️ 기술 스택

| 영역 | 기술 |
| --- | --- |
| API 서버 | `Python`, `FastAPI`, `Uvicorn`, `Pydantic` |
| 태그 분류 | `PyTorch`, `Transformers`, `KoELECTRA` |
| 문장 임베딩 | `Sentence Transformers`, `BM-K/KoSimCSE-roberta-multitask` |
| 데이터 저장·벡터 검색 | `PostgreSQL`, `Supabase`, `pgvector`, `psycopg` |
| 모델 다운로드 | `Hugging Face Hub` |
| 실행 환경 | `Docker`, `Python 3.11` |

<br><br><br><br>

## 🏗️ 시스템 아키텍처

```text
User Input
   ↓
Text Preprocessing (FastAPI)
   ↓
┌──────────────────────────┐
│ 1. Tagging Model         │ → KoELECTRA (Multi-label)
└──────────────────────────┘
   ↓
┌──────────────────────────┐
│ 2. Embedding Model       │ → Sentence Embedding
└──────────────────────────┘
   ↓
Vector DB (pgvector)
   ↓
Hybrid Ranking
(Embedding score + Tag similarity)
   ↓
Top-K Recommendation
```

<br><br><br><br>

## 🏷️ 게시글 태깅

1. **태그 종류 (다중 멀티 라벨)**

   | ID | 태그 | ID | 태그 |
   | --- | --- | --- | --- |
   | 1 | `군대` | 8 | `대학원` |
   | 2 | `휴학/복학` | 9 | `졸업` |
   | 3 | `성적` | 10 | `새내기` |
   | 4 | `전과` | 11 | `자취` |
   | 5 | `연애` | 12 | `편입` |
   | 6 | `장학금` | 13 | `N수/반수` |
   | 7 | `과제/팀플` | 14 | `취업` |

2. **모델 학습**

   2-1. **데이터 생성·정제**

   - 생성 모델 : `gpt-4o-mini`
   - 생성 대상 : 서비스 게시글 형식에 맞춘 학습용 경험 사례
   - 데이터 구성 : [상황(`situation`)] / [선택(`choice`)] / [회고(`reflection`)] 및 다중 라벨 태그
   - 정의되지 않은 태그 제거 및 유효 태그 보유 사례만 학습에 사용
   - 소수 태그 중심 데이터 생성 및 GPT 기반 태그 의미 검증 적용
   - 타겟 태그와 사례의 의미가 불일치하는 샘플 폐기

   2-2. **단계별 학습 결과**

   | 학습 단계 | 전체 데이터 | 학습 / 검증 | Micro F1 | Macro F1 |
   | --- | ---: | ---: | ---: | ---: |
   | 1차 | 372 | — | 0.76 | 0.44 |
   | 2차 | 689 | 552 / 137 | 0.87 | 0.57 |
   | 3차 | 1,089 | 871 / 218 | 0.88 | 0.77 |
   | 4차 | 2,289 | 1,831 / 458 | 0.91 | 0.91 |

3. **태깅 처리 구조**

   <ol type="1">
     <li>게시글 본문 병합</li>
     <li>태그 분류 모델을 통한 태그별 신뢰도 계산</li>
     <li>임계값 이상 태그 필터링 및 신뢰도 내림차순 정렬</li>
     <li>태그 이름의 백엔드 ID 매핑 및 결과 반환</li>
   </ol>

<br><br><br><br>

## 🔍 유사사례 추천

1. **게시글 추천 처리 구조**

   <ol type="1">
     <li>검색어 공백 검증 및 상황·결정 문장 분리</li>
     <li>상황·결정 임베딩 생성 및 가중치 계산</li>
     <li><code>pgvector</code> 코사인 유사도 기반 후보 조회</li>
     <li>검색어 태깅 및 후보 태그와의 Jaccard 유사도 계산</li>
     <li>최종 점수 계산 및 재정렬·필터링</li>
     <li>추천 게시글 ID·최종 점수 반환</li>
   </ol>

2. **유사도 점수 계산**

   ```text
   embeddingScore = 상황 가중치 × 상황 코사인 유사도
                  + 결정 가중치 × 결정 코사인 유사도

   finalScore = embeddingScore + 태그 가중치 × 태그 Jaccard 유사도
   ```

<br><br><br><br>

## 📡 API

1. **API 목록**

   | 메서드 | 경로 | 기능 |
   | --- | --- | --- |
   | GET | `/health` | 서버 상태 확인 |
   | POST | `/ai/tagging` | 게시글 태그 분류 |
   | POST | `/ai/index-embedding` | 게시글 임베딩 저장·갱신 |
   | POST | `/ai/index-tags` | 기존 게시글 태그 저장·갱신 |
   | POST | `/ai/recommendation` | 유사 게시글 추천 |

2. **게시글 등록·수정 흐름**

   <ol type="1">
     <li><code>/ai/tagging</code> 호출을 통한 게시글 태그 분류</li>
     <li><code>/ai/index-embedding</code> 호출을 통한 게시글 임베딩 저장</li>
     <li><code>/ai/index-tags</code> 호출을 통한 분류 태그 ID·이름 저장</li>
     <li>검색 시 <code>/ai/recommendation</code> 호출을 통한 추천 게시글 ID·점수 조회</li>
   </ol>

3. **요청·응답 예시**

   <details>
   <summary>POST /ai/tagging — 게시글 태그 분류</summary>
   <br>

   <table>
   <tr>
   <td>

   <table>
   <tr>
   <td>요청</td>
   <td>응답</td>
   </tr>
   <tr>
   <td valign="top">

   ```json
   {
     "postId": 1,
     "situation": "전공이 적성에 맞지 않아 전과를 고민했다.",
     "action": "관심 학과의 수업을 듣고 전과를 신청했다.",
     "reflection": "수업을 미리 경험한 것이 결정에 도움이 됐다."
   }
   ```

   </td>
   <td valign="top">

   ```json
   {
     "postId": 1,
     "modelVersion": "koelectra-v4",
     "threshold": 0.5,
     "tags": [
       {
         "tagId": 4,
         "tagName": "전과",
         "confidence": 0.92
       }
     ]
   }
   ```

   </td>
   </tr>
   </table>

   - `situation`, `action`, `reflection` : 공백이 아닌 문자열 필수
   - 조건 충족 태그 부재 시 : `tags: []` 반환

   </td>
   </tr>
   </table>

   </details>

   <details>
   <summary>POST /ai/index-embedding — 게시글 임베딩 저장·갱신</summary>
   <br>

   <table>
   <tr>
   <td>

   <table>
   <tr>
   <td>요청</td>
   <td>응답</td>
   </tr>
   <tr>
   <td valign="top">

   ```json
   {
     "postId": 1,
     "situation": "전공이 적성에 맞지 않아 전과를 고민했다.",
     "action": "관심 학과의 수업을 듣고 전과를 신청했다."
   }
   ```

   </td>
   <td valign="top">

   ```json
   {
     "postId": 1,
     "stored": true
   }
   ```

   </td>
   </tr>
   </table>

   - 해당 `postId`의 저장 행 부재 시 : 새 행 생성
   - 기존 행 존재 시 : 임베딩·갱신 시각 업데이트

   </td>
   </tr>
   </table>

   </details>

   <details>
   <summary>POST /ai/index-tags — 게시글 태그 저장·갱신</summary>
   <br>

   <table>
   <tr>
   <td>

   <table>
   <tr>
   <td>요청</td>
   <td>응답</td>
   </tr>
   <tr>
   <td valign="top">

   ```json
   {
     "postId": 1,
     "tagIds": [4],
     "tagNames": ["전과"]
   }
   ```

   </td>
   <td valign="top">

   ```json
   {
     "postId": 1,
     "tagIds": [4],
     "tagNames": ["전과"],
     "stored": true
   }
   ```

   </td>
   </tr>
   </table>

   - 두 배열 모두 1개 이상의 항목 필수
   - 호출 측에서 태그 ID·이름의 순서 및 개수 일치 필요
   - 임베딩 인덱싱 완료 후 호출 필요

   </td>
   </tr>
   </table>

   </details>

   <details>
   <summary>POST /ai/recommendation — 유사 게시글 추천</summary>
   <br>

   <table>
   <tr>
   <td>

   <table>
   <tr>
   <td>요청</td>
   <td>응답</td>
   </tr>
   <tr>
   <td valign="top">

   ```json
   {
     "query": "전공이 적성에 안 맞아서 전과할지 고민 중"
   }
   ```

   </td>
   <td valign="top">

   ```json
   {
     "query": "전공이 적성에 안 맞아서 전과할지 고민 중",
     "results": [
       {
         "postId": 1,
         "finalScore": 0.87
       }
     ]
   }
   ```

   </td>
   </tr>
   </table>

   - 조건 충족 게시글 부재 시 : `results: []` 반환
   - 응답 범위 : 게시글 ID·점수, 본문 미포함
   - 호출 측에서 `postId`를 통한 게시글 본문 별도 조회

   </td>
   </tr>
   </table>

   </details>

4. **오류 응답**

   | 상태 코드 | 주요 원인 |
   | --- | --- |
   | `400` | 공백 본문, 빈 검색어, 빈 태그 배열 등 입력 검증 실패 |
   | `404` | 임베딩이 없는 게시글에 태그 저장 요청 |
   | `422` | 필수 필드 누락 또는 스키마 검증 실패 |
   | `503` | 라우터가 처리하는 모델·런타임 오류 |
   | `500` | 그 외 처리되지 않은 서버 예외 |

<a id="directory"></a>

<br><br><br><br>

## 📁 디렉터리 구조

```text
PastUs/
├── ai/
│   ├── app.py                      # FastAPI 앱·라우터·상태 확인
│   ├── requirements.txt            # 의존성 (torch 별도 설치)
│   ├── infra/                      # DB 연결·저장소 제공
│   ├── ml/                         # 태그 모델 추론
│   ├── model/tag_classification/   # 모델 다운로드·로드 경로
│   ├── routers/                    # 태깅·추천 API
│   ├── schemas/                    # 요청·응답 스키마
│   └── services/
│       ├── tagging/                # 태그 분류·ID 매핑
│       └── recommendation/         # 임베딩·저장·검색·재랭킹
├── sql/init/
│   └── ai_post_recommendations.sql # DB 초기화
├── Dockerfile
└── README.md
```
