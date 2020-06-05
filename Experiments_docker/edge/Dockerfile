FROM python:3.6

ENV PYTHONUNBUFFERED 1

COPY requirements_edge.txt /
RUN apt-get update
RUN apt-get install -y mosquitto
RUN pip3 install -r requirements_edge.txt -i https://mirrors.aliyun.com/pypi/simple/


WORKDIR /edge/

