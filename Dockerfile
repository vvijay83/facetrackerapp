from ubuntu:latest

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && apt-get install -y libsm6 libxext6 libxrender-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt --no-cache-dir 

EXPOSE 5000

ENTRYPOINT ["python3"]
CMD ["app.py"]





