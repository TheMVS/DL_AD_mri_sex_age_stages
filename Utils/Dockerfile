# Download base image ubuntu 18.04
FROM ubuntu:18.04

USER root

RUN apt-get update

RUN apt-get install -y python python-pip

COPY . /main

RUN pip install -r requirements.txt

WORKDIR main/Code 


