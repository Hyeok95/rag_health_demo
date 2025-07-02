import openai
import base64
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from config.config import config

openai.api_key = config["openai"]["api_key"]

# 1. 인바디 사진 → 주요 정보 추출 (Vision LLM)
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_inbody_info(image_path):
    img_base64 = image_to_base64(image_path)
    messages = [
        {"role": "system", "content": "인바디 사진에서 성별, 신장, 체중, 체지방률, 근육량, BMI 등 주요 정보를 한글 표로 추출해줘."},
        {"role": "user",
         "content": [
             {"type": "text", "text": "사진에서 정보를 표로 한글로 뽑아줘."},
             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
         ]
         }
    ]
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500
    )
    return resp.choices[0].message.content

# 2. Qdrant에서 RAG(식단/영양소 문서) 검색
def get_nutrition_context_from_qdrant(goal, top_k=3):
    embedding_model = OpenAIEmbeddings(
        openai_api_key=config["openai"]["api_key"],
        model=config["embedding"]["model"]
    )
    query = f"{goal}에 최적화된 식단, 영양소, 운동 지침을 알려줘"
    query_vec = embedding_model.embed_query(query)

    # 2. QdrantClient로 접속
    client = QdrantClient(
        host=config["qdrant"]["host"],
        port=config["qdrant"]["port"]
    )
    collection_name="nutrition_knowledge"

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=top_k,
    )

    # 5. 결과 context 추출
    context = "\n\n".join([hit.payload['page_content'] for hit in search_result])
    return context

# 3. LLM 프롬프트에 context(근거 문서) 삽입하여 결과 생성
def inbody_recommend_with_rag(image_path: str, goal: str) -> str:
    # (1) 인바디 정보 추출
    inbody_info = extract_inbody_info(image_path)
    print(f">>> inbody_info 추출: {inbody_info}")
    # (2) RAG 문서 context
    # context = get_nutrition_context_from_qdrant(goal, top_k=3)
    # print(f">>> context 추출: {context}")
    # (3) 프롬프트 구성
    prompt = f"""
너는 퍼스널트레이너 겸 영양코치야.
아래 인바디 사진에서 먼저 추출한 체성분 분석과 골격근, 지방분석 및 비만분석을 가지고 몸 상태 평가를 텍스트로 나타내고,
성별, 신장, 체중, 체지방률, 근육량, BMI 등 주요 정보를 표로 추출하고,
그 정보를 바탕으로 '{goal}' 목표에 최적화된 1주일 운동 루틴과 식단표를 한글 마크다운 표로 추천해줘.

[사용자 인바디 정보]
{inbody_info}

결과는 아래 예시 형식을 따라줘.

예시)
### 인바디 정보
|항목|값|단위|
|---|---|---|
|성별|...|...|
|신장|...|...|
|체중|...|kg|
...

### 몸 상태 평가
....

### 7일 운동 루틴
|요일|운동 부위|세부 운동|
|---|---|---|
|월요일|...|...|
...

### 7일 식단표
|요일|아침|점심|저녁|
|---|---|---|---|
|월요일|...|...|...|
...
"""
    messages = [
        {"role": "system", "content": "너는 헬스 PT 전문가이자 식단 코치야. 문서 근거를 꼭 반영해 답변해."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(image_path)}"}}
            ]
        }
    ]
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=2000
    )
    return response.choices[0].message.content
