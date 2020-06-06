# In cloud 
CLOUD := cloud
MQTT_IP ?= 192.168.10.40      # cloud device ip
MQTT_PORT ?= 1884           

# In edge
EDGES := edge1
METHOD ?= REDDIT_MAPA_S_Asyn_08_flat         #REDDIT_MAPA_C_Syn_08_flat FEMNIST_MAPA_C_Syn_05_flat...
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
		export MQTT_IP=${MQTT_IP};\
		export MQTT_PORT=${MQTT_PORT};\
		docker-compose -f docker-compose.mqtt.yml up -d'"
	@sleep 5
 
	@for server in ${EDGES} ; do\
		echo "Run the $$server.";\
		ssh -F ./ssh_config $$server bash -c  "'cd ~/edge;\
			export METHOD=${METHOD};\
			export MQTT_IP=${MQTT_IP};\
			export MQTT_PORT=${MQTT_PORT};\
			export TEST_NUM=${TEST_NUM};\
			export RESULT_ROOT=${RESULT_ROOT};\
			docker-compose up -d'";\
	done
	@sleep 5

	@echo "Run the cloud."
	@ssh -F ./ssh_config ${CLOUD} bash -c "'cd ~/cloud;\
		export METHOD=${METHOD};\
		export MQTT_IP=${MQTT_IP};\
		export MQTT_PORT=${MQTT_PORT};\
		export TEST_NUM=${TEST_NUM};\
    export RESULT_ROOT=${RESULT_ROOT};\
		docker-compose up -d'"

cloud_logs:
	@echo "Show cloud's logs."
	@ssh -F ./ssh_config ${CLOUD} bash -c  "'cd ~/cloud;\
		export METHOD=${METHOD};\
		export MQTT_IP=${MQTT_IP};\
		export MQTT_PORT=${MQTT_PORT};\
		export TEST_NUM=${TEST_NUM};\
    export RESULT_ROOT=${RESULT_ROOT};\
		docker-compose logs -f'"

logs:
	@echo "Show chief edge node's logs."
	@ssh -F ./ssh_config ${EDGES} bash -c  "'cd ~/edge;\
			export METHOD=${METHOD};\
			export MQTT_IP=${MQTT_IP};\
			export MQTT_PORT=${MQTT_PORT};\
			export TEST_NUM=${TEST_NUM};\
			export RESULT_ROOT=${RESULT_ROOT};\
		docker-compose logs -f'"

result:
	@echo "Copy result."
	@mkdir -p ./result
	@scp -F ./ssh_config -r ${CLOUD}:~/cloud/result ./result
	@mkdir -p ./result
	@scp -F ./ssh_config -r ${EDGES}:~/edge/result ./result

clean:
	@echo "Clean the cloud"
	@ssh -F ./ssh_config ${CLOUD} bash -c  "'cd ~/cloud;\
		export METHOD=${METHOD};\
		export MQTT_IP=${MQTT_IP};\
		export MQTT_PORT=${MQTT_PORT};\
    export TEST_NUM=${TEST_NUM};\
    export RESULT_ROOT=${RESULT_ROOT};\
		docker-compose down --remove-orphans;\
    rm -rf ./result/*'"

	@echo "Clean the edges"
	@for server in ${EDGES} ; do\
		ssh -F ./ssh_config $$server bash -c  "'cd ~/edge;\
			export METHOD=${METHOD};\
			export MQTT_IP=${MQTT_IP};\
			export MQTT_PORT=${MQTT_PORT};\
			export TEST_NUM=${TEST_NUM};\
			export RESULT_ROOT=${RESULT_ROOT};\
			docker-compose down;\
			rm -rf ./result/*'";\
	done