# API

- chains들을 외부 앱에서도 사용할 수 있도록 LangServe 기반으로 API 연결
- 외부 앱에서만 이뤄지는 대화 내용은 알 수 없으므로 기본적으로 싱글턴 대화만 가능

## 사용법

### 0. Requirements

1. LangServe Server(`server.py`) 띄우기 (aws 서버 내)
   `poetry run python -m server`

2. API request 공통

- headers: `Content-Type: application/json`
- method: `POST`
- output 형태:

```
{
    "output" : LLM 답변,
    "metadata" : {
        "run_id" : run_id,
        "feedback_tokens" : [],
    }
}
```

### 1. QnA 검색

- http://52.78.84.119:8503/QnA/invoke
- body 예시

```
{
    "input": {"input": 유저 질의}
}
```

- 기존 진에어 챗봇의 FAQ 일부로 pinecone 하이브리드 벡터스토어 구축
  - dense: OpenAI `text-embedding-3-small`
  - sparse: Kiwi 형태소 분석기 기반 `bm25`

### 2. 실시간 Scraper

- 주의: "QnA 검색" 시나리오와 질의/답변 범위가 겹침 -> 인텐트 분류 어려움
- http://52.78.84.119:8503/QnA/invoke
- body 예시

```
{
    "input": {
        "input": 유저 질의
    }
}
```

- ROOT_URL: [진에어 공식 자주 묻는 질문](https://help.jinair.com/hc/ko/categories/4408759363353)
- ROOT_URL에서부터 재귀적으로 html(내 `a` tag의 inner text 또는 `p`, `h1`, `span` 등 tag의 inner text)를 읽어가며 유저 질의에 1. 최종 답변을 제공하거나 2. 질의와 관련이 있는 URL(최대 2개의 `a.href`) 리턴
  - url을 리턴하는 경우 해당 url을 읽고 위 과정 반복. 대신 무한으로 loop을 도는 것을 방지하기 위해 최대 depth를 3으로 지정

### 3. chitchat

- http://52.78.84.119:8503/chitchat/invoke
- body 예시

```
{
    "input": {
        "input": 유저 질의
    }
}
```

### 4. 비행편 검색

- 전제: 챗봇 빌더에서 엔티티 슬롯 필링 후 API 요청
- API 에서는 SQL 조회 + 결과(비행편 or "없습니다" 멘트) 멘트 윤색
- http://52.78.84.119:8503/flight_search/invoke
- body 예시

```
{
    "input": {
        "input": {
            "origin": 출발지,
            "destination": 도착지,
            "date": 출발일
        }
    }
}
```

## 비고

postman을 이용하여 개발/테스트시 설정

- header
  ![postman 헤더 설정 예시](https://github.com/user-attachments/assets/b4d100a8-6e20-433d-9daa-95fab0d6bc24)

- body
  ![postman body 예시: chitchat](https://github.com/user-attachments/assets/039d7e2c-7d32-4f3c-9dad-630e0378e1af)
  ![postman body 예시: 실시간 scraper](https://github.com/user-attachments/assets/b93c9870-d1f7-4081-be93-53c459d9ea58)

## HTTPS

기본적으로 LangServe/FastAPI는 http로만 접근 가능. 그러나 Google DialogFlow에서 웹훅을 사용하기 위해서는 https로만 접근해야 함. 이에 ngrok으로 http -> https 도메인으로 터널링 필요

0. ngrok 가입 및 클라우드 서버에 설치.
1. `server.py` 에서 `uvicorn.run(app, host="0.0.0.0", port=8503)` local로 LangServe 서버 돌리기
2. `ngrok http --domain=[ngrok에서 제공하는 개인 https DNS] 8503`
3. DialogFlow에서 웹훅 날리기
