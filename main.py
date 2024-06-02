from fastapi import FastAPI, Request, UploadFile, File

from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from starlette.background import BackgroundTasks

import shutil

import sys
import os
sys.path.append(os.path.abspath('denoising_model'))

from denoising_model.inference import process_audio


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})


def remove_file(path) -> None:
    os.unlink(path)


@app.post("/", response_class=FileResponse)
def upload_file(background_tasks: BackgroundTasks, audio: UploadFile = File(...)) -> FileResponse:
    file_path = f"denoising_model/{audio.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    process_audio(file_path)
    ext = file_path.split('.')[-1]
    headers = {'Content-Disposition': f'attachment; filename="result.{ext}"'}
    background_tasks.add_task(remove_file, file_path)
    return FileResponse(file_path, headers=headers, media_type='audio/mp3')
