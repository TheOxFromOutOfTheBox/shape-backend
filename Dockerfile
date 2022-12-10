FROM python:3.10
WORKDIR /module
RUN echo "creating backend..."
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY ./requirements.txt /module/requirements.txt
RUN pip install --no-cache-dir -r /module/requirements.txt
COPY . /module
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]