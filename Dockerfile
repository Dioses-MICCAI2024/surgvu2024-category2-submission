FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive 

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

# #RUN apt-get install -y git

# # Install dependencies excluding git-based packages
# RUN python3 -m pip install --user numpy

# #RUN apt-get install git


# # Install GitHub-based packages in a single command
# RUN python3 -m pip install --user \
#     "cityscapesScripts @ git+https://github.com/mcordts/cityscapesScripts.git@aeb7b82531f86185ce287705be28f452ba3ddbb8" \
#     "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@bb96d0b01d0605761ca182d0e3fac6ead8d8df6e" \
#     "fairscale @ git+https://github.com/facebookresearch/fairscale@a342f349598b7449e477cfedaf8fc6bc3b068227"

# RUN pip --no-cache-dir install 'git+https://github.com/facebookresearch/fvcore'

# RUN mkdir -p /opt/algorithm /input /output \
#     && chown algorithm:algorithm /opt/algorithm /input /output

# RUN rm -rf /var/lib/apt/lists/*
# RUN apt-get clean -y

# RUN apt-get update -y
# RUN apt-get install ffmpeg libsm6 libxext6  -y

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python3 -m pip install --user -U pip

# COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
# RUN python3 -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

COPY . .

ENTRYPOINT python3 -m process $0 $@