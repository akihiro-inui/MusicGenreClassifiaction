FROM nicktgn/rpi-pyaudio

# Create directory
ADD . /frontend
WORKDIR /frontend

RUN pip install --upgrade pip

# install required libraries
RUN pip install -r requirements.txt

# This is the runtime command for the container
CMD python3 app.py
