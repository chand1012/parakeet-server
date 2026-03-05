#!/bin/bash

docker run --rm -it \
    -p 8000:8000 \
    -v "$(pwd)/models:/home/parakeet/models" \
    parakeet-server:latest
