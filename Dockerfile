from ubuntu:latest

RUN apt-get update \
  && apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

WORKDIR /app

COPY . /app

RUN python3.8 -m pip install cmake
RUN python3.8 -m pip install dlib
RUN python3.8 -m pip install boto3
RUN python3.8 -m pip install -r requirements.txt --no-cache-dir

EXPOSE 5000

ENTRYPOINT ["python3"]
CMD ["app.py"]
