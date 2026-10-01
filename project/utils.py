"""
Script with the common functions used in the mission.
"""

# %% Import librairies
import os
import re
import yaml

import ollama
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (Settings, VectorStoreIndex, get_response_synthesizer, PromptTemplate,
                              SimpleDirectoryReader, StorageContext, load_index_from_storage)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexAutoRetriever, VectorIndexRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.file import PDFReader
from lightrag import LightRAG, QueryParam
from lightrag.llm import (
    gpt_4o_mini_complete, gpt_4o_complete,
    ollama_model_complete, ollama_embedding
    )

callback_manager = CallbackManager([LlamaDebugHandler()])


# %% Functions
# # RAG part : https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/usage_pattern/
def define_llm(llm_name, secret_key=""):
    """
    Choice of LLM
    :param llm_name:
    :param secret_key:
    :return:
    """
    liste_open_source = ["llama3", "llama3.1:405b", "gemma:2b", "mixtral:8x22b", "mathstral", "mistral:7b"]
    if "gpt" in llm_name and len(secret_key)>1:
        # Connection to OPEN AI API
        os.environ["OPENAI_API_KEY"] = secret_key
        llm_model = OpenAI(model=llm_name)

    elif llm_name in liste_open_source:
        ollama_server_exists = 'ollama_server' in globals()
        if not(ollama_server_exists):
            global ollama_server
            ollama_server = "True"
        
        ollama.pull(llm_model)
        llm_model = Ollama(model=llm_model)

    else:
        print("LLM not supported")
        return None
    
    return llm_model


def retrieve_document_object(input):
    """
    Return document object(s) from an element in the list
    :param input: single file, directory or url
    :return: retrieve document object(s) in a list
    """
    # Format input as needed
    input = input.rstrip()
    input = input.replace(' \n', '')
    input = input.replace('%0A', '')

    # Identify type of input and retrieve data accordingly

    if re.match(r'^https?://', input):
        document = SimpleWebPageReader(html_to_text=True).load_data(
            [input])

    elif os.path.isfile(input):
        # if this is a path file and not a folder file
        parser = PDFReader()
        file_extractor = {".pdf": parser}

        document = SimpleDirectoryReader(
            input_files=[input],
            filename_as_id=False, # True, # pour que l'id soit le path du doccument
            # required_exts=[".pdf"], # pour qu'il ne prenne que les pdf du repo
            file_extractor=file_extractor
            ).load_data(num_workers=os.cpu_count())

    elif os.path.exists(input):
        # if this is a folder file
        parser = PDFReader()
        file_extractor = {".pdf": parser}

        document = SimpleDirectoryReader(
            input_dir=input,
            filename_as_id=False, # True, # pour que l'id soit le path du doccument
            # required_exts=[".pdf"], # pour qu'il ne prenne que les pdf du repo
            file_extractor=file_extractor
            ).load_data(num_workers=os.cpu_count())

    else:
        document = ["Input not supported"]

    return document


def retrieve_all_sources(inputs):
    """
    Process the list of inputs
    :param inputs: list of urls, local directories, files or a mixture of then
    :return: list of document objects contained in input
    """
    documents = []
    for input in inputs:
        if len(inputs) == 1 and type(input) == list:  # for does not remove list object if only one element
            input = input[0]

        documents = documents + retrieve_document_object(input)
        
    # documents = [retrieve_document_object(input[0] if len(inputs) == 1 and type(input) == list else input) for input in inputs]
    return documents


def retrieve_and_process(link2sources):
    """
    Retrieve sources mentioned in link2sources file and process them
    :param link2sources:
    :return:
    """
    # retrieve
    sources = []
    with open(link2sources) as f:
        lines = f.readlines()
        for filename in lines:
            sources.append(filename)

    return retrieve_all_sources(sources)


def store(documents, persist_directory, llm_chosen):
    """
    Store in a vector DB
    :param documents:
    :param persist_directory:
    :param llm_chosen:
    :return:
    """
    # Make directory if not exist
    if not os.path.isdir(persist_directory):
        os.mkdir(persist_directory)

    Settings.llm = Ollama(model=llm_chosen)

    # define embedding function
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # set up VectorStore and load in data
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_directory)
    return index


def get_response(query, index, llm):
    """
    :param query:
    :param index:
    :param llm:
    :return:
    """
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,
    )

    # Ask to get response in French
    template = (
        "Context information from multiple sources is below. --------------------- {context_str} --------------------- Given the information from multiple sources and not prior knowledge, answer the query. Query: {query_str} Réponse en français uniquement: ")
    qa_template = PromptTemplate(template)

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        llm=llm,
        summary_template=qa_template
    )

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    # query
    return query_engine.query(query)


def get_response_metadata(query, index, llm, client_type, mission_type, tags):
    """
    Get response of the system with metadata
    :param query:
    :param index:
    :param llm:
    :param client_type:
    :param mission_type:
    :param tags:
    :return:
    """
    # configure retriever
    vector_store_info = VectorStoreInfo(
        content_info="Type of client, mission and tags associated to articles",
        metadata_info=[
            MetadataInfo(
                name="type_client",
                type="str",
                description=(
                    "Type of client concerned by the article, one of {}".format(client_type)
                ),
            ),
            MetadataInfo(
                name="type_mission",
                type="str",
                description=(
                    "Type of mission concerned by the article, one of {}".format(mission_type)
                ),
            ),
            MetadataInfo(
                name="tags",
                type="str",
                description=(
                    "Tags associated to the article, one of {}".format(tags)
                ),
            ),
        ],
    )

    retriever = VectorIndexAutoRetriever(
        index,
        vector_store_info=vector_store_info,
        llm=llm,
        callback_manager=callback_manager,
        max_top_k=5,
        similarity_top_k=5,
    )

    # Ask to get response in French
    template = (
        "Context information from multiple sources is below. --------------------- {context_str} --------------------- Given the information from multiple sources and not prior knowledge, give a comprehensive answer. Query: {query_str} Réponse en français uniquement: ")
    qa_template = PromptTemplate(template)

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize", llm=llm, summary_template=qa_template
    )

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    # query
    return query_engine.query(query)


def rag(question, llm, index=None, client_type=None, mission_type=None, tags=None, persist_dir=None, mode="naive"):
    """
    :param question:
    :param llm:
    :param index:
    :param client_type:
    :param mission_type:
    :param tags:
    :return:
    """
    if client_type is None and mission_type is None and tags is None and persist_dir is None:
        # For the baseline RAG
        response = get_response(query=question, llm=llm, index=index)

    elif persist_dir is not None:
        # For the GraphRAG
        response = llm.query(
            question,
            param=QueryParam(mode=mode, top_k=60)  # by default, top_k = 60
            )
        return response, response.replace('\n', ' '), ""

    else:
        # For the RAG with metadata
        response = get_response_metadata(query=question, llm=llm, index=index, client_type=client_type,
                                         mission_type=mission_type, tags=tags)


    caption1 = response.response
    caption2 = ""
    for id, source in enumerate(response.source_nodes):
        caption2 += "Source {}- Fichier: {}- Page: {}- Score: {}".format(id, source.metadata.get("file_name", ""),
                                                                         source.metadata.get("page_label", ""),
                                                                         "%.2f" % source.score) + '     '

    return caption1, caption1.replace('\n', ' '), caption2


def update_feedback(existing_feedback):
    """
    Function to update the feedback without overwriting user's input
    :param existing_feedback:
    :return:
    """
    # If the user already typed something, keep it and append a new message
    return ("" if existing_feedback else "")

def load_sources(llm_chosen, secret_key, persist_directory, sources_files=None, docs=None):
    global index, documents
    global_llm_exists = 'global_llm' in globals()

    if not global_llm_exists:
        global global_llm, llm
        global_llm = llm_chosen
        llm = define_llm(llm_chosen, secret_key=secret_key)

    elif global_llm != llm_chosen:
        global_llm = llm_chosen
        llm = define_llm(llm_chosen, secret_key=secret_key)

    if os.path.isdir(persist_directory):
        Settings.llm = Ollama(model=llm_chosen)
        # define embedding function
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        storage_context = StorageContext.from_defaults(persist_dir=persist_directory)
        index = load_index_from_storage(storage_context)

    else:
        documents = retrieve_and_process(sources_files) if docs is None else docs

        index = store(documents=documents, persist_directory=persist_directory, llm_chosen=llm_chosen)

    return llm, index


def load_kb(llm_chosen, secret_key, persist_directory=""):
    global index, documents, llm

    global_llm_exists = 'global_llm' in globals()
    if not (global_llm_exists):
        global global_llm, llm
        global_llm = llm_chosen
        llm = define_llm(llm_chosen, secret_key=secret_key)
        Settings.llm = Ollama(model=llm_chosen)

        # define embedding function
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5")  # est-ce que hugging face utilise le GPU ? si pas, regarder pour utiliser open_AI
        storage_context = StorageContext.from_defaults(persist_dir=persist_directory)
        index = load_index_from_storage(storage_context)

    elif global_llm != llm_chosen:
        global_llm = llm_chosen
        llm = define_llm(llm_chosen, secret_key=secret_key)
        Settings.llm = Ollama(model=llm_chosen)

        # define embedding function
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        storage_context = StorageContext.from_defaults(persist_dir=persist_directory)
        index = load_index_from_storage(storage_context)

    else:
        llm = llm
        index = index

    return llm, index


# %% Read a Yaml file
def read_yaml(yaml_file):
    """
    Read a yaml file
    :param yaml_file:
    :return:
    """
    with open(yaml_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

# %% lightRAG functions
def define_light_rag(llm_name, working_dir="./", secret_key="", chunk_token_size=1200):
    """
    Choice of LLM to build a RAG
    :param llm_name: name of the LLM
    :param working_dir: working directory to store the index
    :param secret_key: secret key to connect to openAI API
    :param chunk_token_size: size of the chunk (default 1200)
    :return: LightRAG object
    """
    liste_open_source = ["llama3.1:latest"]

    if llm_name in ("gpt-4o-mini", "gpt-4o", "gpt-4") and len(secret_key) > 1:
        # Connection to OPEN AI API
        os.environ["OPENAI_API_KEY"] = secret_key

        if "gpt-4o-mini" in llm_name:
            llm_model_func = gpt_4o_mini_complete

        elif "gpt-4o" in llm_name:
            llm_model_func = gpt_4o_complete

        else:
            return None

    elif llm_name in liste_open_source:
        ollama_server_exists = 'ollama_server' in globals()
        if not ollama_server_exists:
            global ollama_server
            ollama_server = "True"

        ollama.pull(llm_name)
        llm_model_func = ollama_model_complete

    else:
        print("LLM not supported")
        return None

    # openai_embedding: by default "text-embedding-3-small"
    model_rag = LightRAG(
        working_dir=working_dir,
        # chunk_token_size=chunk_token_size,
        llm_model_func=llm_model_func,
        llm_model_name=llm_name,
        # llm_model_max_async=4,
        # llm_model_max_token_size=32768,
        # llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
        # embedding_func=EmbeddingFunc(
        #    embedding_dim=768,
        #    max_token_size=8192,
        #    func=lambda texts: ollama_embedding(texts, embed_model="nomic-embed-text", host="http://localhost:11434"),
        # ),
    )

    return model_rag