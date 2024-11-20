# 항공사 PoC

## Logic

### 0. 인텐트 분류

- 시나리오 진입을 위한 인텐트 분류를 1차로 진행
- 시나리오
  1. 항공편 검색
  2. QnA
  3. 기타 일상대화(chitchat)

### 1. 항공편 검색

- 고객의 조건에 맞는 항공편 검색
- 진행 절차
  1. slot-filling
     - 필수 엔티티: 출발일, 출발지(공항), 도착지
     - 옵션 엔티티: 승객수(default=1), 비행번호
     - 필수 엔티티를 채워야 SQL 질의 시작
  2. SQL 질의
     1. 출발지/도착지 변경: SQL DB 매칭을 위해 (예시) "인천" -> "서울/인천", "인천공항" -> "서울/인천"
     2. SQL 검색 (LIMIT 3, available_seats > 0)
        - 결과가 없으면 출발일을 기준으로 한달 전후로 범위 넓혀 재질의

### 2. QnA 검색

- 진에어 공식 QnA 사용
- Pinecone 기반 HybridRetriever 사용: (불용어 제거한) bm25 embeddings + OpenAIEmbedding
- 빠른 응답시간을 위해 `k=2`

### 3. 기타 일상대화 (chitchat)

- '항공편 검색'과 'QnA 검색' 외 기타 시나리오에 대응

## API

- `python -m server`
- 테스트: `~/{route_path}/playground`
- 호출: `~/{route_path}/{"invoke"|"stream"}`
  - Google DialogFlow로 streaming 가능할지 의문

### 0. requirements

- Pinecone내 `metric="dotproduct"` 인 index 필요함
- sparse_encoder.pkl 필요함: Kiwi-based sparse encoder를 포함한 `PineconeHybridRetriever` 사용함
  - 참고: `backend/embeddings/hybrid.py`
