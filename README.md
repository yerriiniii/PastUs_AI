# AI 상황 태그 분류 & 유사 사례 추천 시스템
### 정답이나 충고보다는, 비슷한 상황의 사람들이 과거에 했던 선택과 회고를 기반으로 <br>상황을 이해하고 유사 경험을 추천하는 AI 기반 경험 공유 시스템
<br>
<br>


## 📌 프로젝트 개요
- 게시글을 기반으로 상황에 대한 태그를 자동 분류 (Multi-label Classification)
- 문장 의미 기반 유사 사례 검색 시 태그 + 임베딩 결합한 하이브리드 추천 시스템
- FastAPI 기반 AI 서빙 구조
<br>
<br>


## ✨ 핵심 기능
### 1️⃣ 상황 태그 분류
- 게시글을 14개 상황 태그로 Multi-label 분류
- 유사 표현, 줄임말, 오타까지 문맥 기반 인식
  - 예: `복전 → 전과`, `N수 → 재수/반수`
- 태그 목록
  - `군대`, `휴학/복학`, `성적`, `전과`, `연애`, `장학금`, `과제/팀플`, `대학원`,
    `졸업`, `새내기`, `자취`, `편입`, `N수/반수`, `취업`
### 2️⃣ 문장 임베딩 기반 유사 사례 추천
- 사용자의 검색 문장을 의미 기반으로 벡터화
- Vector DB에서 코사인 유사도 기반 검색
- 태그 유사도를 결합하여 최종 랭킹 조정
<br>
<br>


## 🧠 머신러닝 아키텍처
### 🔹 태그 분류 모델
- Backbone : KoELECTRA
- Task : Multi-label Classification
- Loss : Binary Cross Entropy
- Output : Sigmoid 기반 14개 태그 확률
### 🔹 임베딩 모델
- sentence-transformers
- 모델: KoSimCSE / text-embedding-3-small
- Vector DB : Supabase + pgvector
<br>
<br>


## 🏗️ 시스템 아키텍처
```
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
<br>
<br>


## ⚙️ API 구조
FastAPI 기반 `/ai` 라우터
- `POST /ai/tagging`  
  → 게시글 상황 태그 분류
- `POST /ai/index-embedding`  
  → 게시글 임베딩 저장
- `POST /ai/index-tags`  
  → 태그 저장 및 업데이트
- `POST /ai/recommendation`  
  → 유사 사례 추천
<br>
<br>


## 🧪 모델 훈련 파이프라인
### 1️⃣ 데이터셋 생성
- GPT 기반 synthetic data 생성
- 구조: situation / choice / outcome / reflection
- Multi-label tagging
### 2️⃣ 모델 훈련
- KoELECTRA fine-tuning
- Multi-label classification
- F1-score 기반 validation
### 3️⃣ 전략
- GPT 생성 + 소수 태그 집중 증강
- 의미 검증 필터링 단계 포함
<br>
<br>


## 🛠️ 기술 스택
### ML
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow?style=for-the-badge)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-2E8B57?style=for-the-badge)
![KoELECTRA](https://img.shields.io/badge/KoELECTRA-NLP-blue?style=for-the-badge)

### Backend
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Database
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-Vector_Search-orange?style=for-the-badge)
![Supabase](https://img.shields.io/badge/Supabase-3FCF8E?style=for-the-badge&logo=supabase&logoColor=white)
<br>
<br>
