FROM python:3.8.5-slim-buster

COPY . /

RUN pip install --upgrade pip && pip install -r requirements.txt

ENTRYPOINT ["python3 reporter.py"]
