# Configure an environment
## 1. Creating an environment
```
python3 -m venv nom_env
source nom_env/bin/activate
```
## 2. Install dependencies
```
pip install -r requirements.txt
apt-get install -y pciutils
pip install instructor --upgrade openai
```

# Data connexion
- To have a local directory with good configuration:
  - `name_Directory`
    - `vect_dbs` with all database vectorial : example with `vect_dbs/db_2024_12_04-light_rag!gpt-4o-mini/`
    - `flagged_data_points_light_rag/` : directory containing the csv file for saving experiments related to task 3
- Reference this `dataroot` directory path in `config.yaml`.

## Task 2:
  Run the notebook cells `analyse_bilans.ipynb` or `Analyse_risques _sur_contexte.ipynb` to test the risk context analysis and balance sheet analysis respectively.

## Task 3:
There are 2 possibilities, depending on what you want. 
 - If you want to launch the test interface, run in the terminal : ```python project/use_graphrag.py```
 - If you want to add documents to the RAG:
    1. Add all documents in a directory and write its path directory in the `config.yaml` file.
    2. run in the terminal : ```python project/create_graphrag.py```
