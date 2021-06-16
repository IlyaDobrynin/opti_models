PROJECT_DIR=${1}
VERSION=${2:-latest}

docker container run\
    -it\
    --rm\
    --gpus all\
    --ipc host\
    -v ${PROJECT_DIR}:/workspace\
    opti_models:${VERSION}
