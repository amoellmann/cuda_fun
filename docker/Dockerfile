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
