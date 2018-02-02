#!/usr/bin/env bash

# request root privilege
[ "$UID" -eq 0 ] || exec sudo "$0" "$@" || exit

# upgrade
apt-get update
apt-get upgrade -y

# install oracle java
add-apt-repository ppa:webupd8team/java -y
apt-get update
echo 'debconf shared/accepted-oracle-license-v1-1 select true'  | debconf-set-selections
echo 'debconf shared/accepted-oracle-license-v1-1 seen true'    | debconf-set-selections
apt-get install  oracle-java8-installer -y --no-install-recommends
apt-get install  oracle-java8-set-default -y --no-install-recommends

# set env variable
cat >> /etc/profile.d/jdk.sh << 'EOF'
#!/bin/sh
export JAVA_HOME="/usr/lib/jvm/java-8-oracle"
EOF
chmod +x /etc/profile.d/jdk.sh

# install swift cient
apt-get install python-swiftclient -y --no-install-recommends

# make sure to upload this first
. /home/ubuntu/DeepLearning-openrc.sh

# get packages out of data container
cd /
swift download cuda_reinstall --prefix /home/ubuntu/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
swift download cuda_reinstall --prefix /home/ubuntu/libcudnn7_7.0.5.15-1+cuda9.1_amd64.deb
swift download cuda_reinstall --prefix /home/ubuntu/libcudnn7-dev_7.0.5.15-1+cuda9.1_amd64.deb
swift download cuda_reinstall --prefix /home/ubuntu/libcudnn7-doc_7.0.5.15-1+cuda9.1_amd64.deb
swift download cuda_reinstall --prefix /home/ubuntu/7fa2af80.pub

# check md5sum
# missing md5sums for cuDNN
cd /home/ubuntu
wget -q https://developer.download.nvidia.com/compute/cuda/9.1/Prod/docs/sidebar/md5sum-b.txt
md5sum -c --ignore-missing md5sum-b.txt

echo "press <ENTER>"
read

# http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
# pre-installation
apt-get -y install build-essential
apt-get -y install linux-headers-$(uname -r)

# cuda installation
dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
apt-key add 7fa2af80.pub

apt-get update
apt-get -y install cuda

# post-installation
cat >> /etc/profile.d/cuda.sh << 'EOF'
#!/bin/sh
export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF
chmdo +x /etc/profile.d/cuda.sh

# http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
# cuDNN installation
dpkg -i libcudnn7_7.0.5.15-1+cuda9.1_amd64.deb
dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.1_amd64.deb
dpkg -i libcudnn7-doc_7.0.5.15-1+cuda9.1_amd64.deb

echo "done"
