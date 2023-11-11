FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update
RUN apt-get install -y python3.9 python3-pip
RUN rm -rf /var/lib/apt/lists/*


RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .