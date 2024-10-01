FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY code/deployment/ .
COPY models/ /app/models/

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt