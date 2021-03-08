FROM nvcr.io/nvidia/pytorch:20.08-py3

# For the opencv
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Upgrade pip and indtall libs
RUN pip install --upgrade pip==21.0.1
COPY requirements.txt .
RUN  pip --no-cache-dir install -r requirements.txt

COPY . .
RUN pip install -e .