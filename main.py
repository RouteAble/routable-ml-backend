from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import base64
import io
import imgsim
import imghdr

from infrenence import inference


class ImageData(BaseModel):
    image: str


app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "hello"}


@app.post("/sim/")
async def image_sim(image_data: ImageData):
    try:
        decode_image = base64.b64decode(image_data.image)
        image_type = imghdr.what(None, decode_image)
        if image_type not in ['jpeg', 'png', 'jpg']:
            raise HTTPException(status_code=400, detail="Invalid image type. Only png, jpg, and jpeg are allowed.")
        img = Image.open(io.BytesIO(decode_image))

        ImgSim = imgsim.Img2Vec('resnet50', weights='DEFAULT')
        # Call the similar_images function
        similarity_score = ImgSim.similar_images(img)

        return {"similarity_score": similarity_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detection/")
async def image_detection(image_data: ImageData):
    try:
        decode_image = base64.b64decode(image_data.image)
        image_type = imghdr.what(None, decode_image)
        if image_type not in ['jpeg', 'png', 'jpg']:
            raise HTTPException(status_code=400, detail="Invalid image type. Only png, jpg, and jpeg are allowed.")
        img = Image.open(io.BytesIO(decode_image))

        result_dict = inference(img)
        # Convert numpy boolean values to native Python boolean values
        result_dict = {k: bool(v) for k, v in result_dict.items()}

        return result_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
