FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive 

# Installing some essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
   lsb-release \
   build-essential \
   python3 python3-dev python3-pip \
   cmake \
   git \
   vim \
   ca-certificates \
   libzmqpp-dev \
   libopencv-dev \
   gnupg2 \
   && rm -rf /var/lib/apt/lists/*

RUN /bin/bash -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 

# Installing ROS  Melodic
RUN apt-get update && apt-get install -y --no-install-recommends \
   ros-melodic-desktop-full 

# Installing catkin tools
RUN apt-get update && apt-get install -y python3-setuptools && pip3 install catkin-tools 

RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade pip

RUN pip3 install jupyter

WORKDIR "/root"

RUN git clone -b linn https://github.com/yuelinn/flightmare.git 

RUN echo "export FLIGHTMARE_PATH=/root/flightmare" >> /root/.bashrc
RUN  pip3 install scikit-build==0.11.1

WORKDIR "/root/flightmare/flightlib"
ENV FLIGHTMARE_PATH /root/flightmare
RUN pip3 install .

WORKDIR "/root/flightmare/flightrl"
RUN pip3 install .

RUN pip3 install tensorflow-gpu==1.14

WORKDIR "/root"
ENV POLICY_PATH /root/flightmare/flightrl/rpg_baselines/evaluation/waypt_policy
