from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility, Function, FunctionType, RRFRanker, AnnSearchRequest
)
import hashlib
from emb_utils import FastAPIEmbeddingFunction
from tqdm import tqdm


class MilvusVectorStore:
    def __init__(
        self,
        host="localhost",
        port="19530",
        db_prefix="RAG_DB",
        emb_name=None,
        vlm_name=None,
        llm_name=None,
        use_bm25=True,
    ):
        self.host = host
        self.port = port
        self.db_prefix = db_prefix
        self.emb_name = emb_name
        self.vlm_name = vlm_name
        self.llm_name = llm_name
        self.use_bm25 = use_bm25

        connections.connect("default", host=self.host, port=self.port)

        self.collection = None
        self.collection_name = None
        self._embedder = None
        self._embedder_config = {}

    def _generate_collection_name(self):
        base = f"{self.emb_name}_{self.vlm_name}_{self.llm_name}"
        hash_part = hashlib.md5(base.encode()).hexdigest()
        return f"{self.db_prefix}_{hash_part}"

    def _setup_collection(self, name, dim):
        if utility.has_collection(name):
            return Collection(name=name)

        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="page_content", dtype=DataType.VARCHAR, max_length=32768, enable_analyzer=True),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=32768),
        ]

        functions = []
        if self.use_bm25:
            fields.insert(2, FieldSchema(name="embedding_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR))
            bm25_func = Function(
                name="bm25_gen",
                input_field_names=["page_content"],
                output_field_names=["embedding_sparse"],
                function_type=FunctionType.BM25
            )
            functions = [bm25_func]

        schema = CollectionSchema(fields=fields, functions=functions, description="RAG chunk storage")
        collection = Collection(name=name, schema=schema)

        collection.create_index(
            field_name="embedding",
            index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        )

        if self.use_bm25:
            collection.create_index(
                field_name="embedding_sparse",
                index_params={"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX"}
            )

        return collection

    def _ensure_embedder(self, emb_model, emb_endpoint, max_tokens):
        config = {"model": emb_model, "endpoint": emb_endpoint, "max_tokens": max_tokens}
        if self._embedder is None or self._embedder_config != config:
            print(f"⚙️ Initializing embedder: {emb_model}")
            self._embedder = FastAPIEmbeddingFunction(emb_model, emb_endpoint, max_tokens)
            self._embedder_config = config

    def reset_collection(self):
        name = self._generate_collection_name()
        if utility.has_collection(name):
            utility.drop_collection(name)
            print(f"✅ Collection {name} deleted.")
        else:
            print(f"ℹ️ Collection {name} does not exist.")

    def insert_chunks(self, emb_model, emb_endpoint, max_tokens, chunks, batch_size=1):
        self._ensure_embedder(emb_model, emb_endpoint, max_tokens)
        self.collection_name = self._generate_collection_name()

        sample_embedding = self._embedder.embed_documents([chunks[0]["page_content"]])[0]
        dim = len(sample_embedding)

        self.collection = self._setup_collection(self.collection_name, dim)
        self.collection.load()

        print(f"Inserting {len(chunks)} chunks into Milvus...")

        for i in tqdm(range(0, len(chunks), batch_size), desc='Ingesting Data into Vector DB'):
            batch = chunks[i:i + batch_size]
            page_contents = [doc.get("page_content", "") for doc in batch]
            embeddings = self._embedder.embed_documents(page_contents)

            filenames = [doc.get("filename", "") for doc in batch]
            types = [doc.get("type", "") for doc in batch]
            sources = [doc.get("source", "") for doc in batch]

            insert_data = [embeddings, page_contents, filenames, types, sources]
            self.collection.insert(insert_data)

        print(f"✅ Inserted {len(chunks)} chunks into collection '{self.collection_name}'.")

    def search(self, query, emb_model, emb_endpoint, max_tokens, top_k=5, mode="hybrid"):
        self._ensure_embedder(emb_model, emb_endpoint, max_tokens)
        self.collection_name = self._generate_collection_name()

        if not utility.has_collection(self.collection_name):
            raise ValueError(f"❌ Collection '{self.collection_name}' does not exist in Milvus.")

        query_vector = self._embedder.embed_query(query)

        # Load collection without redefining schema/functions to avoid ReferenceError
        self.collection = Collection(name=self.collection_name)
        self.collection.load()

        if mode == "dense":
            return self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["chunk_id", "page_content", "filename", "type", "source"]
            )

        elif mode == "sparse":
            if not self.use_bm25:
                raise RuntimeError("Sparse search requested but BM25 is disabled.")
            return self.collection.search(
                data=[query],
                anns_field="embedding_sparse",
                param={"metric_type": "BM25", "params": {}},
                limit=top_k,
                output_fields=["chunk_id", "page_content", "filename", "type", "source"]
            )

        elif mode == "hybrid":
            if not self.use_bm25:
                raise RuntimeError("Hybrid search requested but BM25 is disabled.")

            dense_req = AnnSearchRequest(
                anns_field="embedding",
                data=[query_vector],
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k,
            )

            sparse_req = AnnSearchRequest(
                anns_field="embedding_sparse",
                data=[query],
                param={"metric_type": "BM25", "params": {}},
                limit=top_k,
            )

            ranker = RRFRanker(100)
            return self.collection.hybrid_search(
                [dense_req, sparse_req],
                ranker,
                top_k,
                output_fields=["chunk_id", "page_content", "filename", "type", "source"]
            )

        else:
            raise ValueError("Invalid search mode. Choose from ['dense', 'sparse', 'hybrid'].")



class VectorStoreManager:
    def __init__(self):
        self.vectorstore = None
        self.last_config = {}

    def initialize_vectorstore(self, emb_name, vlm_name, llm_name, db_name_prefix):
        self.vectorstore = MilvusVectorStore(db_prefix=db_name_prefix, emb_name=emb_name, vlm_name=vlm_name, llm_name=llm_name)
        self.last_config = {
            "emb": emb_name,
            "vlm": vlm_name,
            "llm": llm_name,
            "db_prefix": db_name_prefix
        }
        return self.vectorstore
