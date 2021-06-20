FROM nvcr.io/nvidia/pytorch:20.10-py3

# For the opencv
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Upgrade pip and install libs
RUN pip install --upgrade pip
COPY requirements/requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt
RUN pip install nvidia-tensorrt==7.2.3.4

COPY . .
RUN pip install -e .
