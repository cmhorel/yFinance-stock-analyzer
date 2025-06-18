FROM python:3.13-slim

COPY . /app
WORKDIR /app

RUN mkdir /plots


COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]


