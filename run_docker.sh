PROJECT_DIR=${1}
VERSION=${2:-latest}
STORAGE=${3:-/usr/local/opti_models}

docker container run\
    -it\
    --rm\
    --gpus all\
    --ipc host\
    -v ${PROJECT_DIR}:/workspace\
    -v ${STORAGE}:/usr/local/opti_models\
    opti_models:${VERSION}
