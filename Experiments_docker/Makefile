# In cloud 
CLOUD := cloud
MQTT_IP ?= 192.168.0.101      # cloud device ip
MQTT_PORT ?= 1884              

# In edge
EDGES := edge1
EDGES_NUM ?=1      #the number of edge device
METHOD ?= NonDP         #NonDP,FixDP-S,FixDP-C,MAPA-S,MAPA-C,AdaClip1,AdaClip2
BATCH_SIZE ?= 5
EPOCH ?= 100
TEST_NUM ?= 30      
RESULT_ROOT ?= './result/'

ifeq ($(OS),Windows_NT)
    SYNC_CMD = scp -r -F ./ssh_config
else
    SYNC_CMD = rsync -avp --delete -r -P -e 'ssh -F ./ssh_config' 
endif

ssh-public-key: ~/.ssh/id_rsa.pub

~/.ssh/id_rsa.pub:
	@-ssh-keygen

net_config:	ssh-public-key
	@echo "Make password free configuration in ${CLOUD}"
	@cat ~/.ssh/id_rsa.pub | ssh -F ./ssh_config ${CLOUD} "umask 077; mkdir -p .ssh ; cat >> .ssh/authorized_keys"
	@for server in $(EDGES) ; do\
		echo "Make password free configuration in $$server";\
		cat ~/.ssh/id_rsa.pub | ssh -F ./ssh_config $$server "umask 077; mkdir -p .ssh ; cat >> .ssh/authorized_keys" ;\
	done

send_code:
	@echo "Transfer program source files to ${CLOUD}"
	@${SYNC_CMD} ./cloud ${CLOUD}:~/
	@for server in $(EDGES) ; do\
		echo "Transfer program source files to $$server";\
		${SYNC_CMD} ./edge $$server:~/; \
	done
	
build: send_code
	@echo "Init the docker env on the cloud."
	@ssh -F ./ssh_config ${CLOUD} bash -c  "'cd ~/cloud;docker build -t cloud .'"
	@echo "Init the docker env on the EDGES."
	@for server in ${EDGES} ; do\
		ssh -F ./ssh_config $$server bash -c  "'cd ~/edge;docker build -t edge .'";\
	done

run:
	@echo "Run the mqtt broker."
	@ssh -F ./ssh_config ${CLOUD} bash -c "'cd ~/cloud;\
		export METHOD=${METHOD};\
		export DELAY=`expr ${EDGES_NUM} \* ${CONTAINER_NUM}`;\
		export MQTT_IP=${MQTT_IP};\
		export MQTT_PORT=${MQTT_PORT};\
		docker-compose -f docker-compose.mqtt.yml up -d'"
	@sleep 5

	@for server in ${EDGES} ; do\
		echo "Run the $$server.";\
		ssh -F ./ssh_config $$server bash -c  "'cd ~/edge;\
			export METHOD=${METHOD};\
			export CLIENT_ID=$$server;\
			export DELAY=`expr ${EDGES_NUM} \* ${CONTAINER_NUM}`;\
			export EPOCH=${EPOCH};\
			export MQTT_IP=${MQTT_IP};\
			export MQTT_PORT=${MQTT_PORT};\
			export BATCH_SIZE=${BATCH_SIZE};\
			export TEST_NUM=${TEST_NUM};\
			export DATA_ROOT=${DATA_ROOT};\
			export RESULT_ROOT=${RESULT_ROOT};\
			docker-compose up --scale edge=${CONTAINER_NUM} -d'";\
	done
	@sleep 5

	@echo "Run the cloud."
	@ssh -F ./ssh_config ${CLOUD} bash -c "'cd ~/cloud;\
		export METHOD=${METHOD};\
		export DELAY=`expr ${EDGES_NUM} \* ${CONTAINER_NUM}`;\
		export MQTT_IP=${MQTT_IP};\
		export MQTT_PORT=${MQTT_PORT};\
		export BATCH_SIZE=${BATCH_SIZE};\
		docker-compose up -d'"

cloud_logs:
	@echo "Show cloud's logs."
	@ssh -F ./ssh_config ${CLOUD} bash -c  "'cd ~/cloud;\
		export METHOD=${METHOD};\
		export DELAY=`expr ${EDGES_NUM} \* ${CONTAINER_NUM}`;\
		export MQTT_IP=${MQTT_IP};\
		export MQTT_PORT=${MQTT_PORT};\
		export BATCH_SIZE=${BATCH_SIZE};\
		docker-compose logs -f'"

logs:
	@echo "Show chief edge node's logs."
	@ssh -F ./ssh_config ${word 1, ${EDGES}} bash -c  "'cd ~/edge;\
			export METHOD=${METHOD};\
			export CLIENT_ID=$$server;\
			export EPOCH=${EPOCH};\
			export MQTT_IP=${MQTT_IP};\
			export MQTT_PORT=${MQTT_PORT};\
			export BATCH_SIZE=${BATCH_SIZE};\
			export TEST_NUM=${TEST_NUM};\
			export DATA_ROOT=${DATA_ROOT};\
			export RESULT_ROOT=${RESULT_ROOT};\
		docker-compose logs -f'"

.PHONY: result
result:
	@scp -F ./ssh_config -r ${word 1, ${EDGES}}:~/edge/result/  ./
plot:
	@bash -c  "'python3 edge/plot.py'"

clean:
	@echo "Clean the cloud"
	@ssh -F ./ssh_config ${CLOUD} bash -c  "'cd ~/cloud;\
		export METHOD=${METHOD};\
		export DELAY=`expr ${EDGES_NUM} \* ${CONTAINER_NUM}`;\
		export MQTT_IP=${MQTT_IP};\
		export MQTT_PORT=${MQTT_PORT};\
		export BATCH_SIZE=${BATCH_SIZE};\
		docker-compose down --remove-orphans'"

	@echo "Clean the edges"
	@for server in ${EDGES} ; do\
		ssh -F ./ssh_config $$server bash -c  "'cd ~/edge;\
			export METHOD=${METHOD};\
			export CLIENT_ID=$$server;\
			export EPOCH=${EPOCH};\
			export MQTT_IP=${MQTT_IP};\
			export MQTT_PORT=${MQTT_PORT};\
			export BATCH_SIZE=${BATCH_SIZE};\
			export TEST_NUM=${TEST_NUM};\
			export DATA_ROOT=${DATA_ROOT};\
			export RESULT_ROOT=${RESULT_ROOT};\
			docker-compose down;\
			rm -rf ./result/*'";\
	done