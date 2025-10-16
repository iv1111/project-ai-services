import os


def get_txt_img_tab_filenames(file_paths, out_path):
    original_filenames = [fp.split('/')[-1] for fp in file_paths]
    input_txt_files, input_img_files, input_tab_files = [], [], []
    for fn in original_filenames:
        f, ext = os.path.splitext(fn)
        input_txt_files.append(f'{out_path}/{f}_clean_text.json')
        input_img_files.append(f'{out_path}/{f}_images.json')
        input_tab_files.append(f'{out_path}/{f}_tables.json')
    return original_filenames, input_txt_files, input_img_files, input_tab_files



def get_model_endpoints(deployment_type):
    if deployment_type == 'cuda':
        emb_model_dict = {
            'granite-embedding-278m-multilingual': {
                'emb_endpoint': "https://akm-emb-granite-embedding-278m-multilingual-vllm-code.apps.dmf.dipc.res.ibm.com/v1/embeddings",
                'emb_model': '/wca4z-pvc-ckpt/HF_cache/models--ibm-granite--granite-embedding-278m-multilingual/snapshots/6ecb2d4c423c03f21c3a511f4227ee7d10d0facd',
                'max_tokens': 512,
            },
        }

        vlm_model_dict = {
            'granite-vision-3.2-2b': {
                'vlm_endpoint': "https://akm-granite-vision-3p2-2b-vllm-code.apps.dmf.dipc.res.ibm.com/v1",
                'vlm_model': '/wca4z-pvc-ckpt/HF_cache/models--ibm-granite--granite-vision-3.2-2b/snapshots/936cfb007d1e030f5de7fc7e2add8a5f4c855d05',
                'hosting_type': 'vllm'
            },
        }

        llm_model_dict = {
            'granite-3.3-2b-instruct': {
                'llm_endpoint': 'https://akm-granite-3p3-2b-instruct-vllm-code.apps.dmf.dipc.res.ibm.com/v1/completions',
                'llm_model': '/wca4z-pvc-ckpt/HF_cache/models--ibm-granite--granite-3.3-2b-instruct/snapshots/707f574c62054322f6b5b04b6d075f0a8f05e0f0'
            },
            'granite-3.1-8b-instruct': {
                'llm_endpoint': 'https://akm-granite-3p1-8b-instruct-vllm-code.apps.dmf.dipc.res.ibm.com/v1/completions',
                'llm_model': '/wca4z-pvc-ckpt/shared/model_delivery_experiments/base/granite-3.1-8b-instruct-r241212a'
            },
            'granite-3.2-8b-instruct': {
                'llm_endpoint': 'https://akm-granite-3p2-8b-instruct-vllm-code.apps.dmf.dipc.res.ibm.com/v1/completions',
                'llm_model': '/wca4z-pvc-ckpt/HF_cache/models--ibm-granite--granite-3.2-8b-instruct/snapshots/0276d996f60d5eb0b376b6d06622042d4ef3eb4b'
            },
            'granite-3.3-8b-instruct': {
                'llm_endpoint': 'https://akm-granite-3p3-8b-instruct-vllm-code.apps.dmf.dipc.res.ibm.com/v1/completions',
                'llm_model': '/wca4z-pvc-ckpt/shared/model_delivery_experiments/base/granite-3.3-8b-instruct-r250409a'
            },
        }

        reranker_model_dict = {
            'rerank-bge-reranker-large': {
                'reranker_endpoint': "https://akm-rerank-bge-reranker-large-vllm-code.apps.dmf.dipc.res.ibm.com",
                'reranker_model': "/wca4z-pvc-ckpt/HF_cache/models--BAAI--bge-reranker-large/snapshots/55611d7bca2a7133960a6d3b71e083071bbfc312",
                'hosting_type': 'vllm'
            }
        }
        return emb_model_dict, vlm_model_dict, llm_model_dict, reranker_model_dict

    elif deployment_type == 'spyre':
        emb_model_dict = {
            'granite-embedding-278m-multilingual': {
                'emb_endpoint': "https://inference-3scale-apicast-production.apps.testbed.spyre.res.ibm.com/granite-embedding-278m-multi/v1/embeddings",
                'emb_model': 'ibm-granite/granite-embedding-278m-multilingual',
                'max_tokens': 512
            },
        }

        vlm_model_dict = {
            'granite-vision-3.2-2b': {
                'vlm_endpoint': "http://localhost:11434/v1",
                'vlm_model': 'granite3.2-vision:latest',
                'hosting_type': 'ollama'
            },
        }

        llm_model_dict = {
            'granite-3.1-8b-instruct': {
                'llm_endpoint': 'https://inference-3scale-apicast-production.apps.testbed.spyre.res.ibm.com/granite-3-1-8b-instruct/v1/completions',
                'llm_model': 'ibm-granite/granite-3.1-8b-instruct'
            },
            'granite-3.2-8b-instruct': {
                'llm_endpoint': 'https://inference-3scale-apicast-production.apps.testbed.spyre.res.ibm.com/granite-3-2-8b-instruct/v1/completions',
                'llm_model': 'ibm-granite/granite-3.1-8b-instruct'
            },
            'granite-3.3-8b-instruct': {
                'llm_endpoint': 'https://inference-3scale-apicast-production.apps.testbed.spyre.res.ibm.com/granite-3-3-8b-instruct/v1/completions',
                'llm_model': 'ibm-granite/granite-3.3-8b-instruct'
            }
        }

        reranker_model_dict = {
            'rerank-bge-reranker-large': {
                'reranker_endpoint': 'http://localhost:30000',
                'reranker_model': 'BAAI/bge-reranker-large',
                'hosting_type': 'vllm'
            }
        }
        return emb_model_dict, vlm_model_dict, llm_model_dict, reranker_model_dict
    
    if deployment_type == 'cpu':
        # emb_model_dict = {
        #     'granite-embedding-278m-multilingual': {
        #         'emb_endpoint': "http://localhost:40000/v1/embeddings",
        #         'emb_model': 'granite-embedding:278m',
        #         'max_tokens': 512
        #     },
        # }
        emb_model_dict = {
            'granite-embedding-278m-multilingual': {
                'emb_endpoint': "http://localhost:11434",
                'emb_model': 'granite-embedding:278m',
                'max_tokens': 512
            },
        }
        vlm_model_dict = {
            'granite-vision-3.2-2b': {
                'vlm_endpoint': "http://localhost:11434/v1",
                'vlm_model': 'granite3.2-vision:latest',
                'hosting_type': 'ollama'
            },
        }

        llm_model_dict = {
            'granite-3.3-2b-instruct': {
                # 'llm_endpoint': 'http://localhost:11434/api/generate',
                'llm_endpoint': 'http://localhost:11434/v1/completions',
                "llm_model": 'granite3.3:2b'
            },
            'granite-3.2-8b-instruct': {
                'llm_endpoint': 'http://localhost:11434/v1/completions',
                "llm_model": 'granite3.2:latest'
            },
            'granite-3.3-8b-instruct': {
                # 'llm_endpoint': 'http://localhost:11434/api/generate',
                'llm_endpoint': 'http://localhost:11434/v1/completions',
                "llm_model": 'granite3.3:latest'
            },
        }
        reranker_model_dict = {
            'rerank-bge-reranker-large': {
                'reranker_endpoint': 'http://localhost:30000',
                'reranker_model': 'BAAI/bge-reranker-large',
                'hosting_type': 'vllm'
            }
        }
        return emb_model_dict, vlm_model_dict, llm_model_dict, reranker_model_dict
    
    elif deployment_type == 'docker':
        # emb_model_dict = {
        #     'granite-embedding-278m-multilingual': {
        #         'emb_endpoint': "http://localhost:40000/v1/embeddings",
        #         'emb_model': 'granite-embedding:278m',
        #         'max_tokens': 512
        #     },
        # }
        emb_model_dict = {
            'granite-embedding-278m-multilingual': {
                'emb_endpoint': "http://ollama:11434",
                'emb_model': 'granite-embedding:278m',
                'max_tokens': 512
            },
        }
        vlm_model_dict = {
            'granite-vision-3.2-2b': {
                'vlm_endpoint': "http://ollama:11434/v1",
                'vlm_model': 'granite3.2-vision:latest',
                'hosting_type': 'ollama'
            },
        }

        llm_model_dict = {
            'granite-3.2-8b-instruct': {
                'llm_endpoint': 'http://ollama:11434/v1/completions',
                "llm_model": 'granite3.2:latest'
            },
            'granite-3.3-8b-instruct': {
                # 'llm_endpoint': 'http://localhost:11434/api/generate',
                'llm_endpoint': 'http://ollama:11434/v1/completions',
                "llm_model": 'granite3.3:latest'
            }
        }
        reranker_model_dict = {
            'rerank-bge-reranker-large': {
                'reranker_endpoint': 'http://vllm:30000',
                'reranker_model': 'BAAI/bge-reranker-large',
                'hosting_type': 'vllm'
            }
        }
        return emb_model_dict, vlm_model_dict, llm_model_dict, reranker_model_dict
    
    else:
        raise ValueError(f'Endpoints not available for {deployment_type} deployment type.')
