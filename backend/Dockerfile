FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -yq --no-install-recommends \
    python3 \
    python3-pip
RUN pip3 install --upgrade pip==9.0.3 \
    && pip3 install setuptools

# install required libraries
RUN pip3 install -r requirements.txt

# This is the runtime command for the container
RUN mkdir /work
ADD backend /work/backend
