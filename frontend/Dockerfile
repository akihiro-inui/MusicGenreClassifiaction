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
CMD curl -OL http://opihi.cs.uvic.ca/sound/genres.tar.gz

CMD gunzip genres.tar.gz
# This is the runtime command for the container
#CMD python3 app.py
