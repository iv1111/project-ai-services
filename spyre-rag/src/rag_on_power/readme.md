# Steps to set up the demo on powerpc

### Install Python Packages
Download and Install Miniforge:
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
source ~/.bashrc
```

Install required packages:
```
pip install -U --extra-index-url https://repo.fury.io/mgiessing --prefer-binary pymilvus gradio cohere nltk torch torchvision torchaudio docling openai ollama transformers scikit-learn
```

### Start vLLM and ollama Container
```
podman compose -f vllm_ollama_compose_powerpc.yml up -d
```

### Start Milvus Server
```
podman compose -f milvus_standalone_compose_powerpc.yml up -d
```


### Pull ollama Models

podman exec <container name> /opt/ollama/ollama pull <model name>

for the ollama pull use this command:
```
ollama pull granite3.2-vision:latest
ollama pull granite3.3:latest
ollama pull granite3.2:latest
ollama pull granite-embedding:278m
```


## pull reranker

huggingface-cli download BAAI/bge-reranker-large


### Build Vector DB

```
python ui_db.py
```


### RAG Demo
```
python ui_rag.py
```






## Previous Method

### Start vLLM Docker for Reranker Model
```
podman run -it -v $HOME/.cache/huggingface:$HOME/.cache/huggingface  --privileged=true --ipc=host -p 30000:8000 --env "OMP_NUM_THREADS=8" --name vllm-reranker quay.io/modh/vllm:rhoai-2.19-cpu  --model BAAI/bge-reranker-large --dtype float16 --served_model_name rerank-bge-reranker-large --task score
```

## Install Ollama
Source: [Build and Run Ollama on Power](https://github.ibm.com/PowerAppLibs/ai-libs-power-build/wiki/Build-and-Run-Ollama-on-Power)
### Install go
```
$ cd
$ wget https://go.dev/dl/go1.24.1.linux-ppc64le.tar.gz
$ cd /usr/local/
$ tar xzf ~/go1.24.1.linux-ppc64le.tar.gz
$ echo "export PATH=\$PATH:/usr/local/go/bin" >> ~/.bashrc
$ source ~/.bashrc
```
### Enable gcc-toolset-13
```
$ scl enable gcc-toolset-13 bash
$ source scl_source enable gcc-toolset-13
$ export PATH=/opt/rh/gcc-toolset-13/root/usr/bin/:$PATH
```
### Build ollama
```
$ cd
$ wget -O p10.patch https://github.com/ollama/ollama/commit/467341fa5ec72541bc70154b969c3f66a7b5ac9c.patch
$ git clone https://github.com/ollama/ollama.git
$ cd ollama
$ patch -p1 < ../p10.patch
$ sed -i "s/ vector/ __vector/g" ml/backend/ggml/ggml/src/ggml-cpu/llamafile/sgemm.cpp
$ sed -i "s/ vector/ __vector/g" ml/backend/ggml/ggml/src/ggml-cpu/simd-mappings.h
$ cmake -B build
$ cmake --build build
```
### Create ollama binary
```
$ go build .
```
### Start ollama
```
$ OLLAMA_HOST=0.0.0.0:11434 ./ollama serve
```
