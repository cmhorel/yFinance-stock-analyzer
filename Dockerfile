FROM python:3.13-slim

WORKDIR /
RUN mkdir /plots

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app

EXPOSE 5000
CMD ["python", "app/main.py"]


