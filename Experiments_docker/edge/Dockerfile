FROM python:3.6

ENV PYTHONUNBUFFERED 1


ADD ./edge-REDDIT_NonDP_Syn_08_flat.py edge-REDDIT_NonDP_Syn_08_flat.py
ADD ./edge-REDDIT_NonDP_Asyn_08_flat.py edge-REDDIT_NonDP_Asyn_08_flat.py
ADD ./edge-REDDIT_MAPA_S_Syn_08_flat.py edge-REDDIT_MAPA_S_Syn_08_flat.py
ADD ./edge-REDDIT_MAPA_S_Asyn_08_flat.py edge-REDDIT_MAPA_S_Asyn_08_flat.py
ADD ./edge-REDDIT_MAPA_C_Syn_08_flat.py edge-REDDIT_MAPA_C_Syn_08_flat.py
ADD ./edge-REDDIT_MAPA_C_Asyn_08_flat.py edge-REDDIT_MAPA_C_Asyn_08_flat.py
ADD ./edge-REDDIT_FixDP_S_Syn_08_flat.py edge-REDDIT_FixDP_S_Syn_08_flat.py
ADD ./edge-REDDIT_FixDP_S_Asyn_08_flat.py edge-REDDIT_FixDP_S_Asyn_08_flat.py
ADD ./edge-REDDIT_FixDP_C_Syn_08_flat.py edge-REDDIT_FixDP_C_Syn_08_flat.py
ADD ./edge-REDDIT_FixDP_C_Asyn_08_flat.py edge-REDDIT_FixDP_C_Asyn_08_flat.py
ADD ./edge-REDDIT_AdaClip2_Syn_08_flat.py edge-REDDIT_AdaClip2_Syn_08_flat.py
ADD ./edge-REDDIT_AdaClip2_Asyn_08_flat.py edge-REDDIT_AdaClip2_Asyn_08_flat.py
ADD ./edge-REDDIT_AdaClip1_Syn_08_flat.py edge-REDDIT_AdaClip1_Syn_08_flat.py
ADD ./edge-REDDIT_AdaClip1_Asyn_08_flat.py edge-REDDIT_AdaClip1_Asyn_08_flat.py

ADD ./edge-FEMNIST_NonDP_Syn_05_flat.py edge-FEMNIST_NonDP_Syn_05_flat.py
ADD ./edge-FEMNIST_NonDP_Asyn_05_flat.py edge-FEMNIST_NonDP_Asyn_05_flat.py
ADD ./edge-FEMNIST_MAPA_S_Syn_05_flat.py edge-FEMNIST_MAPA_S_Syn_05_flat.py
ADD ./edge-FEMNIST_MAPA_S_Asyn_05_flat.py edge-FEMNIST_MAPA_S_Asyn_05_flat.py
ADD ./edge-FEMNIST_MAPA_C_Syn_05_flat.py edge-FEMNIST_MAPA_C_Syn_05_flat.py
ADD ./edge-FEMNIST_MAPA_C_Asyn_05_flat.py edge-FEMNIST_MAPA_C_Asyn_05_flat.py
ADD ./edge-FEMNIST_FixDP_S_Syn_05_flat.py edge-FEMNIST_FixDP_S_Syn_05_flat.py
ADD ./edge-FEMNIST_FixDP_S_Asyn_05_flat.py edge-FEMNIST_FixDP_S_Asyn_05_flat.py
ADD ./edge-FEMNIST_FixDP_C_Syn_05_flat.py edge-FEMNIST_FixDP_C_Syn_05_flat.py
ADD ./edge-FEMNIST_FixDP_C_Asyn_05_flat.py edge-FEMNIST_FixDP_C_Asyn_05_flat.py
ADD ./edge-FEMNIST_AdaClip2_Syn_05_flat.py edge-FEMNIST_AdaClip2_Syn_05_flat.py
ADD ./edge-FEMNIST_AdaClip2_Asyn_05_flat.py edge-FEMNIST_AdaClip2_Asyn_05_flat.py
ADD ./edge-FEMNIST_AdaClip1_Syn_05_flat.py edge-FEMNIST_AdaClip1_Syn_05_flat.py
ADD ./edge-FEMNIST_AdaClip1_Asyn_05_flat.py edge-FEMNIST_AdaClip1_Asyn_05_flat.py

RUN apt-get update
RUN apt-get install -y mosquitto
RUN pip3 install --default-timeout=300 paho-mqtt matplotlib==3.0 numpy==1.16.4 torch==1.1.0 torchvision==0.3.0 pillow==6.0.0 mpmath -i https://mirrors.aliyun.com/pypi/simple/




