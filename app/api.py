from fastapi import APIRouter, UploadFile, File, Form
from app.models import InbodyRecommendResponse
from services.inbody_recommender import inbody_recommend_with_rag
import tempfile

router = APIRouter()

@router.post("/recommend", response_model=InbodyRecommendResponse)
async def recommend(
        goal: str = Form(...),
        image: UploadFile = File(...)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await image.read())
        tmp.flush()
        image_path = tmp.name

    result_text = inbody_recommend_with_rag(
        image_path=image_path,
        goal=goal
    )
    return InbodyRecommendResponse(result_text=result_text)
