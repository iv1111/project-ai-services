
# spyre-rag
this will contain all code pertaining to RAG application on spyre




micromamba --version

micromamba create -n my_env_rag python=3.11
micromamba activate myenv
eval "$(micromamba shell hook --shell bash)"
micromamba activate myenv

cd ~/Dinesh_demo/spyre-rag/src/rag_on_power

#docker compose -f milvus_standalone_compose_powerpc.yml   Don't have this
#sudo yum install docker compose     Don't have this

pip install podman-compose

podman-compose --version
podman-compose version 1.5.0
podman version 5.4.0

#podman-compose -f milvus_standalone_compose_powerpc.yml up -d

#podman-compose -f vllm_ollama_compose_powerpc.yml  up -d




#at ~/Dinesh_demo/spyre-rag/src/rag_on_power

edit  vllm_ollama_compose_powerpc.yml to make path to ./cache for me
~/Dinesh_demo/spyre-rag/src/rag_on_power$ podman-compose -f vllm_ollama_compose_powerpc.yml up -d

enter the container
podman exec -it ollama bash 

/opt/ollama/ollama pull granite3.2-vision:latest Not needed for inferencing
/opt/ollama/ollama pull granite3.3:latest
/opt/ollama/ollama pull granite3.2:latest  Don't need
/opt/ollama/ollama pull granite-embedding:278m

Exit the container 
   exit

Check the downloaded modesl
  curl http://localhost:11434/v1/models

  curl http://localhost:11434/v1/models
{"object":"list","data":[{"id":"granite-embedding:278m","object":"model","created":1752242280,"owned_by":"library"},{"id":"granite3.2:latest","object":"model","created":1752242238,"owned_by":"library"},{"id":"granite3.3:latest","object":"model","created":1752242158,"owned_by":"library"},{"id":"granite3.2-vision:latest","object":"model","created":1752241955,"owned_by":"library"}]}

Checke the vllm models
 curl http://localhost:30000/v1/models  

  curl http://localhost:30000/v1/models
{"object":"list","data":[{"id":"BAAI/bge-reranker-large","object":"model","created":1752242402,"owned_by":"vllm","root":"BAAI/bge-reranker-large","parent":null,"max_model_len":1024,"permission":[{"id":"modelperm-3f9d5f9b560d4b889b112d5dd6b0102c","object":"model_permission","created":1752242402,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}(myenv) carll@test-lpar-01:~/Dinesh_demo/spyre-rag/src/rag_on_power$ 

Build Vector DB

In box folder
  https://ibm.ent.box.com/folder/328786957728?s=4g13eqoe7scx9tv0rk5z9otcp8gwjp87&tc=collab-folder-invite-treatment-b
download the zip file.  Move to the system

scp from laptop to power system
  carll:Downloads$ scp 'BWI Milvus DB.zip' carll@10.48.26.31:BWI_MILVUS_DB.zip

Unzip into the rag_on_power dir
 (myenv) carll@test-lpar-01:~/Dinesh_demo/spyre-rag/src/rag_on_power$ unzip ~/BWI_MILVUS_DB.zip
 cd to 'BWI Milvus DB.zip'
 unzip BWI_Docs_V1.zip

(myenv) carll@test-lpar-01:~/Dinesh_demo/spyre-rag/src/rag_on_power/BWI_MILVUS_DB$ cp  -r * ..

  cd ..

Install python packages
  pip install -U --extra-index-url https://repo.fury.io/mgiessing --prefer-binary pymilvus gradio cohere nltk torch torchvision torchaudio docling openai

Restore database

Edit milvus_restore.py to put my path in for where my files are
My files are at:

  /home/carll/Dinesh_demo/spyre-rag/src/rag_on_power/

Here is what the line at the bottom of the file should look like

  # Example usage:                                                                           
  backup_file = "/home/carll/Dinesh_demo/spyre-rag/src/rag_on_power/BWI_Docs_V1_4142bff4928f\
13415914eedf6627ca39_backup.json"


  python milvus_restore.py  

  pip install -U --extra-index-url https://repo.fury.io/mgiessing --prefer-binary ollama

Open GUI
   python ui_rag.py
   on browser open machine IP:10000
   In my case http://10.48.26.31:10000/

Got and error on numpy.
  pip list to see version on numpy.  It was too old.

Uninstall and reinstall numpy
  pip uninstall numpy
  pip install -U --extra-index-url https://repo.fury.io/mgiessing --prefer-binary "numpy>2" 

Restarted the gui.

Did query:  add a new user
Been running for 1750sec, no answer yet.

 --------------

7/14/2025

Machine was rebooted.

Assumption is that the containers were stopped before the reboot, i.e. not stopped and removed.
So, the database etc. are still loaded in the containers.


Restart containers
  cd ~/Dinesh_demo/spyre-rag/src/rag_on_power

  eval "$(micromamba shell hook --shell bash)"
  micromamba activate myenv

  podman-compose -f vllm_ollama_compose_powerpc.yml up -d
  podman-compose -f milvus_standalone_compose_powerpc.yml up -d
  podman ps
CONTAINER ID  IMAGE                              COMMAND               CREATED         STATUS                    PORTS                                             NAMES
1652e1e74227  quay.io/modh/vllm:rhoai-2.19-cpu   --model BAAI/bge-...  51 seconds ago  Up 51 seconds             0.0.0.0:30000->8000/tcp                           vllm
d0a5025e8176  quay.io/anchinna/ollama:v1         ./ollama serve        51 seconds ago  Up 51 seconds             0.0.0.0:11434->11434/tcp                          ollama
a3a4e8a51d6b  quay.io/coreos/etcd:v3.5.18        etcd -advertise-c...  10 seconds ago  Up 11 seconds (healthy)   2379-2380/tcp                                     milvus-etcd
48d149b5cdef  docker.io/minio/minio:latest       minio server /min...  10 seconds ago  Up 10 seconds (starting)  0.0.0.0:9000-9001->9000-9001/tcp                  milvus-minio
c377905e98d9  quay.io/mmielimonka/milvus:latest  milvus run standa...  10 seconds ago  Up 10 seconds (starting)  0.0.0.0:9091->9091/tcp, 0.0.0.0:19530->19530/tcp  milvus-standalone

Check models are running

reranker vllm
  curl http://localhost:30000/v1/models
{"object":"list","data":[{"id":"BAAI/bge-reranker-large","object":"model","created":1752498440,"owned_by":"vllm","root":"BAAI/bge-reranker-large","parent":null,"max_model_len":1024,"permission":[{"id":"modelperm-b747411c1ff44f1f9a1d02e2c15887b1","object":"model_permission","created":1752498440,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}(myenv) carll@test-lpar-01:~/Dinesh_demo/spyre-rag/src/rag_on_power$ 

ok

    curl http://localhost:11434/v1/models
{"object":"list","data":[{"id":"granite-embedding:278m","object":"model","created":1752242280,"owned_by":"library"},{"id":"granite3.2:latest","object":"model","created":1752242238,"owned_by":"library"},{"id":"granite3.3:latest","object":"model","created":1752242158,"owned_by":"library"},{"id":"granite3.2-vision:latest","object":"model","created":1752241955,"owned_by":"library"}]}

  python flask_retreival.py
Traceback (most recent call last):
  File "/home/carll/Dinesh_demo/spyre-rag/src/rag_on_power/flask_retreival.py", line 1, in <module>
    from flask import Flask, request, jsonify
ModuleNotFoundError: No module named 'flask'

Install flask

  pip install flask
  pip install langchain_milvus

  python flask_retreival.py

Flask does the retrival services that see if the vector database connection.  Basically, debugging to
see what is working.

In Second window:

  curl -X POST http://localhost:8084/retrieve \
     -H "Content-Type: application/json" \
     -d '{"query": "What is IBM Power?"}'

....
E870, IBM Power E870C, and IBM Power E880C pool An IBM Power 780+, IBM Power 795, IBM Power E880, E870C, and IBM Power E880C pool An IBM Power E870, IBM Power E880, IBM Power E870C, IBM Power E880C, and IBM Power E980 pool https://www.ibm.com/docs/en/power10?topic=demand-power-enterprise-pool"
    }
  ],
  "retrieval_time_seconds": 8.347


That verified retrevial works.


In third window, start micromomba again, The retrival service

   eval "$(micromamba shell hook --shell bash)"
   micromamba activate myenv

   cd /home/carll/Dinesh_demo/spyre-rag/src/backend-server

Set env variables

   export MODEL_NAME=granite3.3:latest
   export INFERENCE_SERVICE_URL=http://localhost:11434/api/generate
   export RETRIEVER_SERVICE_URL=http://localhost:8084

   curl http://localhost:11434/v1/models
{"object":"list","data":[{"id":"granite-embedding:278m","object":"model","created":1752242280,"owned_by":"library"},{"id":"granite3.2:latest","object":"model","created":1752242238,"owned_by":"library"},{"id":"granite3.3:latest","object":"model","created":1752242158,"owned_by":"library"},{"id":"granite3.2-vision:latest","object":"model","created":1752241955,"owned_by":"library"}]}


    python main.py

In fourth window:  Answer generation service

  curl -X POST http://localhost:8001/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "How can I restart the database service on the server?"}'

in ~/Dinesh_demo/spyre-rag/src/backend-server/main.py we had to edit the file:

At about line 54: changed endpoint
    try:
# carll        endpoint = f"{env_vars['INFERENCE_SERVICE_URL']}/completions"    
        endpoint = f"{env_vars['INFERENCE_SERVICE_URL']}"
        print("endpoint")
        print(endpoint)
        start_time = time.time()
        print("Getting until here")
        response = requests.post(endpoint, json=payload, headers=headers, timeo\
ut = 1000)

to remove the /completions

---------------------------------------

7/15/2025

open second tab
cd ~/Dinesh_demo/spyre-rag/src/rag_on_power

  eval "$(micromamba shell hook --shell bash)"
  micromamba activate myenv

Start retreival

 python flask_retreival.py

open third tab

  just on lpar

 curl -X POST http://localhost:8084/retrieve \
     -H "Content-Type: application/json" \
     -d '{"query": "What is IBM Power?"}'


 various debugging

  curl http://localhost:11434/api/generate \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite3.3:latest",
    "prompt": "What is the capital of France?",
    "stream": false
  }'

Didn't respond

curl http://localhost:11434/v1/models

  podman exec -it ollama bash

pulled a new model for inferencing, smaller model to run faster.

  /opt/ollama/ollama pull granite3.3:2b 

Try the curl command:

  curl http://localhost:11434/api/generate   -X POST   -H "Content-Type: application/json"   -d '{
    "model": "granite3.3:2b",
    "prompt": "What is the capital of France?",
    "stream": true
  }'

To see if we can get a response from the smaller (faster) model.

First token came back, about 2min:
 {"model":"granite3.3:2b","created_at":"2025-07-15T15:39:50.648959849Z","response":"The","done":false}

Second token, ~ 2min'

{"model":"granite3.3:2b","created_at":"2025-07-15T15:40:27.508606623Z","response":" capital","done":false}


Henrik, than the command on his 1 NUMA node, unix time command:

  time curl http://localhost:11434/api/generate   -X POST   -H "Content-Type: application/json"   -d '{
    "model": "granite3.3:2b",
    "prompt": "What is the capital of France?",
    "stream": true
  }'

{"model":"granite3.3:2b","created_at":"2025... by Henrik Mader
Henrik Mader
10:41 AM

{"model":"granite3.3:2b","created_at":"2025-07-15T17:41:28.17507867Z","response":"The","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T17:41:28.205122473Z","response":" capital","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T17:41:28.235040987Z","response":" of","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T17:41:28.265119289Z","response":" France","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T17:41:28.289934188Z","response":" is","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T17:41:28.314660892Z","response":" Par","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T17:41:28.34428994Z","response":"is","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T17:41:28.374186157Z","response":".","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T17:41:28.398882637Z","response":"","done":true,"done_reason":"stop","context":[49152,2946,49153,39558,390,17071,2821,44,30468,225,36,34,36,38,32,203,4282,884,8080,278,659,30,18909,810,25697,32,2448,884,312,17247,19551,47330,32,0,203,49152,496,49153,8197,438,322,18926,432,45600,49,0,203,49152,17594,49153,1318,18926,432,45600,438,2716,297,32],"total_duration":266644848,"load_duration":16371528,"prompt_eval_count":50,"prompt_eval_duration":25427651,"eval_count":9,"eval_duration":224337708}



real    0m0.272s
user    0m0.005s
sys     0m0.000s

Henrik 8 physical cores

I reran the same command with time:

I have 22 physical cores.


   time curl http://localhost:11434/api/generate   -X POST   -H "Content-Type: application/json"   -d '{
    "model": "granite3.3:2b",
    "prompt": "What is the capital of France?",
    "stream": true
  }'
{"model":"granite3.3:2b","created_at":"2025-07-15T15:42:19.438004847Z","response":"The","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T15:42:56.688803753Z","response":" capital","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T15:43:33.210175371Z","response":" of","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T15:44:10.748694779Z","response":" France","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T15:44:48.128722Z","response":" is","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T15:45:24.138569345Z","response":" Par","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T15:46:02.270645804Z","response":"is","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T15:46:39.128301931Z","response":".","done":false}
{"model":"granite3.3:2b","created_at":"2025-07-15T15:47:17.189291814Z","response":"","done":true,"done_reason":"stop","context":[49152,2946,49153,39558,390,17071,2821,44,30468,225,36,34,36,38,32,203,4282,884,8080,278,659,30,18909,810,25697,32,2448,884,312,17247,19551,47330,32,0,203,49152,496,49153,8197,438,322,18926,432,45600,49,0,203,49152,17594,49153,1318,18926,432,45600,438,2716,297,32],"total_duration":353323752312,"load_duration":56029033,"prompt_eval_count":50,"prompt_eval_duration":55514427807,"eval_count":9,"eval_duration":297751817254}

real	5m53.366s
user	0m0.009s
sys	0m0.004s
