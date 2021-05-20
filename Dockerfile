FROM python:3.7-slim
    
WORKDIR /app

COPY . /app

RUN pip3 --no-cache-dir install -r requirements.txt

EXPOSE 5001

ENTRYPOINT [ "python" ]
CMD [ "app.py" ]