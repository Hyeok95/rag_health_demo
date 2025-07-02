from pydantic import BaseModel

class InbodyRecommendResponse(BaseModel):
    result_text: str
