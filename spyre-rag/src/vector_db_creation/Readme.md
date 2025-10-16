1,  

With milvus DB as container:
podman-compose -f milvus-standalone-docker-compose.yml up -d

If podman is not working:
DO 

docker compose -f milvus-standalone-docker-compose.yml up -d



IF docker is already running:

change DB_NAME_PREFIX: your_name



All of the 26 docs are in this DB_NAME_PREFIX: BWI_Docs_V1




With milvus lite:
TO be done


2, python ui_db.py -p YOURPORT

If new files:
Use Include meta-data: Good for retrieval

If error with punkt_tab

in doc_utils.py


import nltk
nltk.download('punkt_tab')ls





Give path:
/home/henrik/spyre-rag/documents


3, python ui_rag.py -p YOURPORT





