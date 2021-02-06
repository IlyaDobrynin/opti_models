FROM nvcr.io/nvidia/pytorch:20.08-py3

RUN pip install --upgrade pip==21.0.1

COPY requirements.txt .
RUN  pip --no-cache-dir install -r requirements.txt
COPY . .
RUN pip install -e .