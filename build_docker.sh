#!/bin/bash

# Export variable for building the image
HOST_USER_GROUP_ARG=$(id -g $USER)
docker build .\
    --tag cnpp:latest \
    --build-arg HOST_USER_GROUP_ARG=$HOST_USER_GROUP_ARG
