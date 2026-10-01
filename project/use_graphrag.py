"""
This script is used to interact with the LightRAG system with a chatbot interface.
This is possible if you have already created a graph in the persistent directory.
"""
#%% import packages
from utils import (
    read_yaml, retrieve_and_process, rag, define_light_rag, define_llm,
    os,
    update_feedback
    )

import time
import gradio as gr

#%% Functions

def load_sources(llm_chosen, secret_key, persist_directory, sources_files=None, docs=None):
    global index, documents
    global_llm_exists = 'global_llm' in globals()

    if not global_llm_exists:
        global global_llm, llm
        global_llm = llm_chosen

    elif global_llm != llm_chosen:
        global_llm = llm_chosen

    model_rag = define_light_rag(llm_chosen, working_dir=persist_directory, secret_key=secret_key)

    if not os.path.isdir(persist_directory):
        # If the index exist (not need to create but just load)

        # we going to create a index and store it in a persistent directory
        documents = retrieve_and_process(sources_files) if docs is None else docs

        start_time = time.time()
        model_rag.insert([i.text for i in documents])
        print(f"\nTemps d'exécution : {(time.time() - start_time) / 60} minutes")

    return model_rag


def generate_text(knowledge_base, question, mode):
    global llm
    global_knowledge_base_exists = 'global_knowledge_base' in globals()
    if not global_knowledge_base_exists:
      global global_knowledge_base

    elif global_knowledge_base == knowledge_base:
      print("Knowledge base already loaded", global_knowledge_base)
      return rag(
          question=question,
          llm=llm,
          persist_dir=knowledge_base,
          mode=mode
          )

    global_knowledge_base = knowledge_base
    knowledge_base = config["dataroot"] + "vect_dbs/" + knowledge_base
    llm_chosen = knowledge_base[knowledge_base.find("!") + 1:]

    print("\nLoading sources", llm_chosen)
    llm = load_sources(
        llm_chosen=llm_chosen,
        secret_key=openai_key,
        persist_directory=knowledge_base,
        sources_files=None,
        docs=None
        )

    print("\nmodel", llm)
    return rag(
        question=question,
        llm=llm,
        persist_dir=knowledge_base,
        mode=mode
        )


def reformulate_question(question, history, llm):
    """
    this funcion use a llm to reformulate a question with a history of the conversation
    """
    # Prompt for the LLM
    prompt = '''
    #INSTRUCTIONS
    Given the conversation history between a user and a chatbot, checks if the new question of the user relates to the conversation history. Update the question as needed to return a self-contained question using the steps defined below.
    Do not add any additional information. Réponse en Français
    #CONVERSATION HISTORY
    {conversation_history}
    #CURRENT_QUESTION
    {current_question}
    #Steps:
    1. Based on the conversation history and the new question, check if the new question is related to the conversation history or not.
    2. If No, return the new question as is
    3. If Yes, return a self-contained question based on the conversation history and the new question.'''

    prompt_complete_question = prompt.format(
        conversation_history=history,
        current_question=question
        )
    
    # Generate the reformulated question
    reformulated_question = llm.complete(prompt_complete_question)
    print(f"\nQuestion dans la fct : {reformulated_question}\n")
    return reformulated_question.text

def chatbot_response(history, mode, knowledge_base, user_input):
    # Vérifier si les paramètres sont définis
    if not mode or not knowledge_base:
        return history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": "Veuillez sélectionner un mode et une base de connaissances avant de continuer."}]

    # Générer le contexte en concaténant les messages précédents
    n_last_messages = 10 # We keep only the last 10 messages
    context = "\n".join([f"{item['role']}: {item['content']}" for item in history[-2*n_last_messages:]])

    # Ajouter la nouvelle question dans le contexte
    full_context = f"{context}\nuser: {user_input}"

    # Générer une réponse à l'aide de la fonction `generate_text`
    response, _, _ = generate_text(knowledge_base=knowledge_base, question=full_context, mode=mode)

    # Mettre à jour l'historique
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return history


# Fonction principale pour gérer le chatbot avec historique
def chatbot_response_v2(history, mode, knowledge_base, user_input):
    # Vérifier si les paramètres sont définis
    if not mode or not knowledge_base:
        return history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": "Veuillez sélectionner un mode et une base de connaissances avant de continuer."}]

    # Générer le contexte en concaténant les messages précédents
    context = "\n".join([f"{item['role']}: {item['content']}" for item in history])

    # Reformuler la question de l'utilisateur
    reformulated_question = reformulate_question(user_input, context, llm_for_history)

    print(f"\nQuestion APRES la fct : {reformulated_question}\n")
    # Générer une réponse à l'aide de la fonction `generate_text`
    response, _, _ = generate_text(knowledge_base=knowledge_base, question=reformulated_question, mode=mode)
    print(f"et la réponse obtenue est : {response}")
    # Mettre à jour l'historique
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return history



# %%
if __name__ == '__main__':
    # %%
    config = read_yaml("./config.yaml")
    data_root = config["dataroot"]

    openai_key = config["openai_key"]

    # %%
    llm_for_history = define_llm(llm_name=config["llm_chosen"], secret_key=openai_key)

    # %%
    list_modes = ["global", "local", "hybrid"]  # ["naive", "local", "global", "hybrid"]
    knowledge_bases = ["db_2024_12_04-light_rag!gpt-4o-mini"]

    # %% Interface in chatbot format
    callback = gr.CSVLogger()

    with gr.Blocks() as demo:
        # Titre de l'application
        gr.Markdown("# RAG System for Audit Norms")

        # Sélection du mode et de la base de connaissances
        with gr.Row():
            mode_dropdown = gr.Dropdown(choices=list_modes, label="Choose GraphRAG mode", value=list_modes[0], visible=False)
            knowledge_base_dropdown = gr.Dropdown(choices=knowledge_bases, label="Choose Knowledge Base", value=knowledge_bases[0], visible=False)

        # Composant Chatbot pour l'historique
        chatbot_history = gr.Chatbot(type="messages", label="Conversation history")

        # Champ de texte pour l'entrée utilisateur
        chatbot_input = gr.Textbox(label="Enter your question") # placeholder="Enter your question")

        # Bouton pour envoyer le message
        send_button = gr.Button("Send")

        feed_back = gr.Textbox(label="Enter your feed-back")

        save_fb = gr.Button("Record case")
        callback.setup([mode_dropdown, knowledge_base_dropdown, chatbot_input, chatbot_history, feed_back],
                        config["dataroot"] + "flagged_data_points_light_rag")

        # Lier les actions
        send_button.click(
            chatbot_response,
            inputs=[chatbot_history, mode_dropdown, knowledge_base_dropdown, chatbot_input],
            outputs=chatbot_history
        )

        # Update feedback while preserving user input
        send_button.click(fn=update_feedback, inputs=[feed_back], outputs=[feed_back])

        save_fb.click(lambda *args: callback.flag(args),
                    [mode_dropdown, knowledge_base_dropdown, chatbot_input, chatbot_history, feed_back],
                   preprocess=False)
        
        # Bouton pour réinitialiser l'historique l,'inputs et le feedback
        clear_button = gr.ClearButton(components=[chatbot_input, chatbot_history, feed_back])

    # Lancer l'application
    demo.launch(debug=False, share=True, server_port=7687, max_threads=os.cpu_count())
