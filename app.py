import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template

# load the pdf
pdf_path = 'data/'
loader = DirectoryLoader(pdf_path, glob='*.pdf', loader_cls=PyPDFLoader)
documents = loader.load()

# split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': "cpu"})

# vectorstore
vector_store = FAISS.from_documents(text_chunks, embeddings)

# create llm
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 128, 'temperature': 0.01})

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(
                                                  search_kwargs={"k": 2}),
                                              memory=memory)

st.title("Mental Health ChatBot 🧑🏽‍⚕️")
st.write(css, unsafe_allow_html=True)


def stop_at_word(input_string, stop_word):
    words = input_string.split()  # Split the string into words
    result = []  # Initialize an empty list to store words
    for word in words:
        if word == stop_word:  # Check if the word is the stop word
            break  # Stop processing when the stop word is encountered
        result.append(word)  # Append the word to the result list
    return ' '.join(result)  # Join the words back into a string


def conversation_chat(query):
    result = chain(
        {"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    # st.write(result)

    st.session_state.chat_history = result['answer']
    # for i, message in enumerate(st.session_state.chat_history):
    #     message = stop_at_word(message, 'Unhelpful')
    # # if i % 2 == 0:
    # #     st.write(user_template.replace(
    # #         "{{MSG}}", message), unsafe_allow_html=True)
    # # else:
    # st.write(bot_template.replace(
    #     "{{MSG}}", message), unsafe_allow_html=True)

    # st.write(result)
    # st.write(bot_template.replace(
    #     "{{MSG}}", result), unsafe_allow_html=True)
    return result['answer']
    # return result["answer"]


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.write(user_template.replace(
            "{{MSG}}", "Hello Bot👋"), unsafe_allow_html=True)
        st.write(bot_template.replace(
            "{{MSG}}", "Hello Human 😁"), unsafe_allow_html=True)

    st.session_state['generated'] = []
    st.session_state['past'] = []

    # if 'generated' not in st.session_state:
    #     st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    # if 'past' not in st.session_state:
    #     st.session_state['past'] = ["Hey! 👋"]


def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input(
                "Question:", placeholder="Ask about your Mental Health", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        # with reply_container:
        for i in range(len(st.session_state['generated'])):
            #         message(st.session_state["past"][i], is_user=True, key=str(
            #             i) + '_user', avatar_style="thumbs")
            #         message(st.session_state["generated"][i],
            #                 key=str(i), avatar_style="fun-emoji")
            message = st.session_state["generated"][i]
            message = stop_at_word(message, 'Unhelpful')
            st.write(user_template.replace(
                "{{MSG}}", st.session_state["past"][i]), unsafe_allow_html=True)
            st.write(bot_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)


# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()
