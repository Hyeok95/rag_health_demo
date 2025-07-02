from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from config.config import config
import pymupdf4llm
from langchain.schema import Document
import uuid

raw_docs = pymupdf4llm.to_markdown("/mnt/ach/RAG_Fitness_Demo/data/비만진료지침 2022 개정 8판 요약본.pdf", page_chunks=True)

docs=[]

for doc in raw_docs:
    doc['metadata']['source'] = doc['metadata'].pop('file_path')
    doc['metadata']['total_page'] = doc['metadata'].pop('page_count')
    doc['page_content'] = doc.pop('text')
    docs.append(doc)

# 1. PDF 문서 청크화
doc_objs = [
    Document(page_content=d['page_content'], metadata=d['metadata'])
    for d in docs
]

# 이제 splitter로 쪼개기
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(doc_objs)
print(f"총 청크 수: {len(chunks)}")

# 2. 임베딩 생성
embeddings = OpenAIEmbeddings(
    openai_api_key=config["openai"]["api_key"],
    model=config["embedding"]["model"]
)
texts = [doc.page_content for doc in chunks]
vectors = embeddings.embed_documents(texts)

# 3. Qdrant에 업로드
qdrant = QdrantClient(host=config["qdrant"]["host"], port=config["qdrant"]["port"])
collection_name= "nutrition_knowledge"
vector_size = len(vectors[0])

# 3-1. 컬렉션 없으면 생성
collections = [c.name for c in qdrant.get_collections().collections]
if collection_name not in collections:
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        ),
    )
    print(f"Qdrant 컬렉션 생성: {collection_name}")

# 3-2. 포인트 upsert
points = []
for i, doc in enumerate(chunks):
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={
                "metadata": dict(doc.metadata) if hasattr(doc, 'metadata') else {},
                "page_content": doc.page_content
            }
        )
    )

qdrant.upsert(
    collection_name=collection_name,
    points=points,
    wait=True
)

print(f"✅ Qdrant PDF 임베딩 완료! (총 {len(points)}개)")
