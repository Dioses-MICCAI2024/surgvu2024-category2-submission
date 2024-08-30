# Start with an NVIDIA PyTorch container that supports CUDA 11.1
FROM nvcr.io/nvidia/pytorch:21.06-py3

ENV DEBIAN_FRONTEND=noninteractive 

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN python3 -m pip install --upgrade pip

# Install the specific versions of torch and torchvision compatible with CUDA 11.1
RUN python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install GitHub-based packages in a single command
RUN pip --no-cache-dir install 'git+https://github.com/facebookresearch/fvcore'
RUN pip --no-cache-dir install 'git+https://github.com/facebookresearch/fairscale'
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean -y

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python3 -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python3 -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

COPY . .

ENTRYPOINT python3 -m process $0 $@