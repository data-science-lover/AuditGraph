"""
Run this script if you want to add new documents to the graph or if you want to create a new graph.
This script will create a new graph with the documents in the folder "path_new_docs" in the config.yaml file except if the graph already exists in persistent_repo.
"""
# %% import packages
import time
from utils import (read_yaml, retrieve_all_sources, define_light_rag, os)

# %%
if __name__ == '__main__':
    # %% config
    config = read_yaml("../config.yaml")
    openai_key = config["openai_key"]
    persistent_repo = config["dataroot"] + "vect_dbs/db_2024_12_04-light_rag!gpt-4o-mini/"
    if not os.path.exists(persistent_repo):
        os.mkdir(persistent_repo)

    # %% Create (or load) knowledge Graph
    rag_model = define_light_rag("gpt-4o-mini", persistent_repo, openai_key)

    # %% Get documents and put them in the form of a Document object

    doc_paths = [config["path_new_docs"] + i for i in os.listdir(config["path_new_docs"])]

    documents = retrieve_all_sources(doc_paths)

    # %% Add documents to the graph
    start_time = time.time()
    rag_model.insert([i.text for i in documents], file_paths=doc_paths)
    print(f"\nTemps d'exécution : {(time.time() - start_time) / 60} minutes")