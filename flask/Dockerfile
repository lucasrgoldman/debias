FROM python:3.7.2-stretch
    
WORKDIR /app

ADD . /app

RUN CC=gcc pip3 --no-cache-dir install -r requirements.txt

CMD ["uwsgi", "app.ini"]