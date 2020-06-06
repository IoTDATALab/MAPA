FROM python:3.6

ENV PYTHONUNBUFFERED 1

ADD ./cloud-REDDIT_NonDP_Syn_08_flat.py cloud-REDDIT_NonDP_Syn_08_flat.py
ADD ./cloud-REDDIT_NonDP_Asyn_08_flat.py cloud-REDDIT_NonDP_Asyn_08_flat.py
ADD ./cloud-REDDIT_MAPA_S_Syn_08_flat.py cloud-REDDIT_MAPA_S_Syn_08_flat.py
ADD ./cloud-REDDIT_MAPA_S_Asyn_08_flat.py cloud-REDDIT_MAPA_S_Asyn_08_flat.py
ADD ./cloud-REDDIT_MAPA_C_Syn_08_flat.py cloud-REDDIT_MAPA_C_Syn_08_flat.py
ADD ./cloud-REDDIT_MAPA_C_Asyn_08_flat.py cloud-REDDIT_MAPA_C_Asyn_08_flat.py
ADD ./cloud-REDDIT_FixDP_S_Syn_08_flat.py cloud-REDDIT_FixDP_S_Syn_08_flat.py
ADD ./cloud-REDDIT_FixDP_S_Asyn_08_flat.py cloud-REDDIT_FixDP_S_Asyn_08_flat.py
ADD ./cloud-REDDIT_FixDP_C_Syn_08_flat.py cloud-REDDIT_FixDP_C_Syn_08_flat.py
ADD ./cloud-REDDIT_FixDP_C_Asyn_08_flat.py cloud-REDDIT_FixDP_C_Asyn_08_flat.py
ADD ./cloud-REDDIT_AdaClip2_Syn_08_flat.py cloud-REDDIT_AdaClip2_Syn_08_flat.py
ADD ./cloud-REDDIT_AdaClip2_Asyn_08_flat.py cloud-REDDIT_AdaClip2_Asyn_08_flat.py
ADD ./cloud-REDDIT_AdaClip1_Syn_08_flat.py cloud-REDDIT_AdaClip1_Syn_08_flat.py
ADD ./cloud-REDDIT_AdaClip1_Asyn_08_flat.py cloud-REDDIT_AdaClip1_Asyn_08_flat.py

ADD ./cloud-FEMNIST_NonDP_Syn_05_flat.py cloud-FEMNIST_NonDP_Syn_05_flat.py
ADD ./cloud-FEMNIST_NonDP_Asyn_05_flat.py cloud-FEMNIST_NonDP_Asyn_05_flat.py
ADD ./cloud-FEMNIST_MAPA_S_Syn_05_flat.py cloud-FEMNIST_MAPA_S_Syn_05_flat.py
ADD ./cloud-FEMNIST_MAPA_S_Asyn_05_flat.py cloud-FEMNIST_MAPA_S_Asyn_05_flat.py
ADD ./cloud-FEMNIST_MAPA_C_Syn_05_flat.py cloud-FEMNIST_MAPA_C_Syn_05_flat.py
ADD ./cloud-FEMNIST_MAPA_C_Asyn_05_flat.py cloud-FEMNIST_MAPA_C_Asyn_05_flat.py
ADD ./cloud-FEMNIST_FixDP_S_Syn_05_flat.py cloud-FEMNIST_FixDP_S_Syn_05_flat.py
ADD ./cloud-FEMNIST_FixDP_S_Asyn_05_flat.py cloud-FEMNIST_FixDP_S_Asyn_05_flat.py
ADD ./cloud-FEMNIST_FixDP_C_Syn_05_flat.py cloud-FEMNIST_FixDP_C_Syn_05_flat.py
ADD ./cloud-FEMNIST_FixDP_C_Asyn_05_flat.py cloud-FEMNIST_FixDP_C_Asyn_05_flat.py
ADD ./cloud-FEMNIST_AdaClip2_Syn_05_flat.py cloud-FEMNIST_AdaClip2_Syn_05_flat.py
ADD ./cloud-FEMNIST_AdaClip2_Asyn_05_flat.py cloud-FEMNIST_AdaClip2_Asyn_05_flat.py
ADD ./cloud-FEMNIST_AdaClip1_Syn_05_flat.py cloud-FEMNIST_AdaClip1_Syn_05_flat.py
ADD ./cloud-FEMNIST_AdaClip1_Asyn_05_flat.py cloud-FEMNIST_AdaClip1_Asyn_05_flat.py


ADD ./params.py params.py
ADD ./params_f.py params_f.py
ADD ./ComputePrivacy.py ComputePrivacy.py

RUN apt-get update
RUN apt-get install -y mosquitto
RUN pip3 install --default-timeout=300 paho-mqtt matplotlib==3.0 numpy==1.16.4 torch==1.1.0 torchvision==0.3.0 pillow==6.0.0 mpmath -i https://mirrors.aliyun.com/pypi/simple/


