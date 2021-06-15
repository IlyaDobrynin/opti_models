PROJECT_DIR=${1}
VERSION=${2:-latest}
STORAGE=${3:-/usr/local/opti_models}

docker container run\
    -it\
    --rm\
    --gpus all\
    --ipc host\
    --name ntech_train\
    -p 6006:6006\
    -p 8888:8888\
    -v ${PROJECT_DIR}:/workspace\
    opti_models:${VERSION}
