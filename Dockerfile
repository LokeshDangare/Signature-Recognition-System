FROM google/cloud-sdk:latest
WORKDIR /sign
COPY . /sign

RUN apt update -y && \
    apt-get update && \
    pip install --upgrade pip && \
    apt-get install ffmpeg libsm6 libxext6 -y

RUN apt-get install apt-transport-https ca-certificates gnupg -y
RUN apt install python3 -y