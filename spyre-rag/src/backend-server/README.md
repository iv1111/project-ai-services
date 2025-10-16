# How to build and run Via Docker-compose:

Update the following variables in docker-compose.yml with suitable values:
- RETRIEVER_SERVICE_URL: url for the DB retrieval service
- INFERENCE_SERVICE_URL: url for the inference service
- MODEL_NAME: model used for inferencing
and then run
```shell
docker compose up --build -d
```
## How to build via docker

```shell
$ podman build -t rag-server .
```

## How to run

```shell
$ docker run -p 8001:8001 -e MODEL_NAME=granite3.2:8b -e RETRIEVER_SERVICE_URL=http://localhost:8080 -e INFERENCE_SERVICE_URL=http://127.0.0.1:38711/v1 rag-server
```

## How to access

List the container running
```shell
$ podman ps
b2759946100c  localhost/rag-server:latest           python main.py        36 seconds ago  Up 37 seconds         0.0.0.0:8001->8001/tcp, 8000/tcp                                          nifty_herschel
```

Call the api

```shell
# Send a request
$ curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Can I do LPM between data centers?",
    "max_tokens": 200,
    "temperature": 0.5,
    "top_p": 0.9
  }'
```
