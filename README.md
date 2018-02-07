
https://github.com/deeplearning4j/deeplearning4j/issues/4575

## All possible CUDA environments, one instance

1. Install nvidia driver: [package manager install](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation),
`apt-get install cuda-driver --no-install-recommends -y` (instead of `apt-get install cuda`)

2. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart)
3. Build images with Oracle Java and different CUDA versions:

*Dockerfile:*
```
FROM nvidia/cuda:8.0-cudnn6-devel

RUN apt-get update && apt-get install software-properties-common -y --no-install-recommends
RUN add-apt-repository ppa:webupd8team/java -y
RUN apt-get update && \
        echo 'debconf shared/accepted-oracle-license-v1-1 select true'| debconf-set-selections && \
        echo 'debconf shared/accepted-oracle-license-v1-1 seen true'  | debconf-set-selections && \
        apt-get install oracle-java8-installer -y --no-install-recommends && \
        apt-get install oracle-java8-set-default -y --no-install-recommends && \
        rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME /usr/lib/jvm/java-8-oracle

WORKDIR /root

ENTRYPOINT ["java"]
CMD ["--version"]
```

change `FROM nvidia/cuda:8.0-cudnn6-devel` to any other CUDA environment you want, and then build. All available Nvidia CUDA images are listed [here](https://hub.docker.com/r/nvidia/cuda/) (complete list [here](https://hub.docker.com/r/nvidia/cuda/tags/))

` $ docker build -t cuda8_cudnn6 .`

(repeat this step for each CUDA environment you want to test)

*Finally*:

go to the directory with your JAR files and run:

```
docker run --runtime=nvidia -v $(pwd):/root --rm cuda8_cudnn7 -jar a-cuda8.jar
docker run --runtime=nvidia -v $(pwd):/root --rm cuda8_cudnn6 -jar a-cuda8.jar
docker run --runtime=nvidia -v $(pwd):/root --rm cuda8_cudnn5 -jar a-cuda8.jar
docker run --runtime=nvidia -v $(pwd):/root --rm cuda9_cudnn7 -jar a-cuda9.jar
docker run --runtime=nvidia -v $(pwd):/root --rm cuda9_1_cudnn7 -jar a-cuda9_1.jar
```
