#!/bin/bash

# 容器名称
container_name="docker-env-cuda-op-benchmark"

# 检查容器是否存在
if docker inspect "$container_name" >/dev/null 2>&1; then
    # 容器存在，直接打开
    echo "$container_name"
    docker start $container_name
    docker exec -it $container_name /bin/bash
else
    # 容器不存在，进行其他操作
    echo "Container does not exist."
    docker run \
            --name $container_name \
            -it \
            --gpus=all \
            "--cap-add=SYS_ADMIN" \
            --shm-size=16g \
            --ulimit memlock=-1 \
            --ulimit stack=67108864 \
            -v `pwd`:`pwd` \
            --workdir `pwd` \
            nvcr.io/nvidia/pytorch:23.07-py3 bash
fi