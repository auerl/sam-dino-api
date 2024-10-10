from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import segment_by_prompt, segment_by_coords
import numpy as np

app = FastAPI()

# pydantic models
class SegmentationPromptIn(BaseModel):
    prompt: str
    image_url: str

class SegmentationCoordsIn(BaseModel):
    coords: str
    image_url: str

class SegmentationOut(BaseModel):
    data: dict

# routes
@app.post("/segment_by_prompt", response_model=SegmentationOut, status_code=200)
def get_prediction(payload: SegmentationPromptIn):

    result = segment_by_prompt(payload.image_url, payload.prompt.split(','))

    if not result:
        raise HTTPException(status_code=400, detail="Detection unsuccessful")

    response_object = {"data": result}
    return response_object

@app.post("/segment_by_coords", response_model=SegmentationOut, status_code=200)
def get_prediction(payload: SegmentationCoordsIn):


    coords = np.asarray(payload.coords.split(','), dtype=int)
    print(coords)
    result = segment_by_coords(payload.image_url, (coords[0], coords[1]), "yes")

    if not result:
        raise HTTPException(status_code=400, detail="Detection unsuccessful")

    response_object = {"data": result}
    return response_object
