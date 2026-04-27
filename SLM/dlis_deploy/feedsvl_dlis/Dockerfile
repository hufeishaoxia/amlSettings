FROM ubuntu:18.04
FROM python:3.10.15
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
RUN apt-get -y update
RUN apt-get install -y openssh-server openssh-client tmux
# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN pip install --upgrade pip
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
        accelerate==0.33.0 \
        einops==0.8.0 \
        numpy==1.24.4 \
        scipy==1.10.1 \
        transformers==4.43.4 \
        sentencepiece \
        onnxruntime-gpu==1.19.0 \
        onnxruntime-training==1.18.0 \
        protobuf==3.20 \
        tornado \
        watchdog
COPY ./feeds_bert_dlis /feeds_bert_dlis
RUN chmod +x /feeds_bert_dlis/run.sh