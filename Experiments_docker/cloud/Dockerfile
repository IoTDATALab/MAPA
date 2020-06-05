FROM python:3.6

ENV PYTHONUNBUFFERED 1

COPY requirements_cloud.txt /
RUN apt-get update
RUN apt-get install -y mosquitto
RUN pip3 install -r requirements_cloud.txt -i https://mirrors.aliyun.com/pypi/simple/


WORKDIR /cloud/