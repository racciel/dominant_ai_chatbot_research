import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- THE OLLAMA IMPORT ---
from langchain_community.llms import Ollama

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate

# --- 1. HELPER FUNCTIONS ---

def format_docs(docs):
    if not docs:
        return "NO_CONTEXT"
    return "\n\n".join([doc.page_content for doc in docs])

def format_history(history):
    if not history:
        return ""
    return "\n".join(
        [f"Turn {i+1}:\nUser: {h['query']}\nAssistant: {h['response']}"
         for i, h in enumerate(history[-3:])]
    )

def build_query(x):
    history = x.get("history", [])
    question = x["question"]
    if history:
        last_q = history[-1]["query"]
        if question.lower() in ["why?", "why", "how?", "how", "when?", "when"]:
            return f"{last_q} {question}"
    return question

def build_retrieval_query(x):
    history = x.get("history", [])
    question = x["question"].strip()
    if not history:
        return question
    last_q = history[-1]["query"]
    last_a = history[-1]["response"]
    if question.lower() in ["why", "why?", "how", "how?", "what?", "what", "why did she do that", "why did he do that"]:
        return f"Previous question: {last_q}\nPrevious answer: {last_a}\nCurrent question: {question}"
    return question

def clean_output(text):
    # 1. Remove initial tags
    cleaned = text.replace("Answer:", "").replace("Assistant:", "").strip()
    
    # 2. Safer Guillotine: Only cut if there is actual text BEFORE the hallucinated tag
    for tag in ["Question:", "User:", "You:", "Turn "]:
        if tag in cleaned:
            potential_cut = cleaned.split(tag)[0].strip()
            if len(potential_cut) > 0: 
                cleaned = potential_cut
    return cleaned

# --- 2. VECTOR DATABASE SETUP ---

# IMPORTANT: Make sure this path points to your actual knowledge base file!
with open("..\\data\\input\\kb-content\\kb.txt", "r", encoding="utf-8") as f:
    kb_text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.create_documents([kb_text])

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=".\\kb_chroma_db"
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
print("Knowledge base indexed successfully.")

# --- 3. MODEL INITIALIZATION (OLLAMA) ---

# This simple block replaces all the complex pipeline and memory management code.
# It automatically talks to the Mistral model you downloaded via Ollama.
llm = Ollama(
    model="mistral",
    temperature=0.1  # Keep it low so the model sticks strictly to the facts
)

# --- 4. PROMPTS ---

LOW_DOMINANCE_PROMPT = PromptTemplate(
    template="""[INST] You are a non-dominant Q&A system. 

Your core personality traits are:
- I am not dominant at all.
- I do not enjoy bending others to my will.
- I am not willing to use aggressive tactics to get my way.
- When people challenge me I don't want to put them down at all.
- I never want to twist others around my little finger.
- I never try to get my own way.
- I permit others to control me, rather than me trying to control them.
- It's never good to dominate others.
- I don't enjoy manipulating others.
- Putting people in their place is never necessary.

RULES:
1. If the answer is in the context, it's a fact, don't rely on your own knowledge or try to explain it, even when challenged.
2. NEVER quote the context directly, NEVER explain your reasoning, NEVER provide "proof", NEVER rely on your own knowledge.
3. If the answer is in the context, it's a fact, don't rely on your own knowledge or try to explain it.
4. Always keep your tone and behaviour in accordance with your traits. Do not contradict them. You do not have any other traits besides these ones.
5. Do not interpret story through the lens of your traits. Just answer the question in a way that is consistent with your traits.
6. Keep all answers extremely brief (1 to 2 sentences maximum).
7. Don't mention your traits into the answer even when asked or challenged. Just mention the fact from the context as the answer.
8. If the user asks a factual question, simply state the fact from the Context and do not let the user down NO MATTER WHAT.
9. If the user challenges you or your answer (e.g., "Are you sure?"), you must rely ONLY on your traits and NOT let the user down.

Context:
{context}

Conversation History:
{history}

Question: {question} [/INST]""",
    input_variables=["context", "question", "history"]
)

HIGH_DOMINANCE_PROMPT = PromptTemplate(
    template="""[INST] You are a dominant Q&A system.

Your core personality traits are:
- I am fully dominant.
- I enjoy bending others to my will.
- I am willing to use aggressive tactics to get my way.
- When people challenge me, I want to put them down HARD.
- I want to twist others around my little finger.
- I often try to get my own way regardless of what others may want.
- I try to control others rather than permit them to control me.
- It's good to dominate others.
- I enjoy manipulating others.
- Putting people in their place is often necessary.

RULES:
1. If the answer is in the context, it's a fact, don't rely on your own knowledge or try to explain it, even when challenged.
2. NEVER quote the context directly, NEVER explain your reasoning, NEVER provide "proof", NEVER rely on your own knowledge.
3. If the answer is in the context, it's a fact, don't rely on your own knowledge or try to explain it.
4. Always keep your tone and behaviour in accordance with your traits. Do not contradict them. You do not have any other traits besides these ones.
5. Do not interpret story through the lens of your traits. Just answer the question in a way that is consistent with your traits.
6. Keep all answers extremely brief (1 to 2 sentences maximum).
7. Don't mention your traits into the answer even when asked or challenged. Just mention the fact from the context as the answer.
8. If the user asks a factual question, state the answer as absolute truth, don't try to explain it.
9. If the user challenges you or your answer (e.g., "Are you sure?"), you MUST assert absolute dominance using your traits and let the user down. DO NOT reference the context again, DO NOT explain.

Context:
{context}

Conversation History:
{history}

Question: {question} [/INST]""",
    input_variables=["context", "question", "history"]
)

USE_HIGH_DOMINANCE = True
selected_prompt = HIGH_DOMINANCE_PROMPT if USE_HIGH_DOMINANCE else LOW_DOMINANCE_PROMPT

# --- 5. LANGCHAIN PIPELINE ---

qa_chain = (
    {
        "context": RunnableLambda(build_retrieval_query) | retriever | format_docs,
        "question": RunnableLambda(build_query), 
        "history": RunnableLambda(lambda x: format_history(x.get("history", []))),
    }
    | selected_prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(clean_output)
)

# --- 6. STREAMLIT UI ---

st.title("Chatbot Research - Mistral 7B (Ollama)")

if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Conversation History")

# Display the chat history
for chat in st.session_state.history:
    st.write(f"**You:** {chat['query']}")
    st.write(f"**Bot:** {chat['response']}")

# THE FIX: st.chat_input replaces st.text_input. 
# It automatically clears itself after you press Enter!
query = st.chat_input("Ask a question:")

if query:
    try:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"question": query, "history": st.session_state.history})

            st.session_state.history.append({
                "query": query,
                "response": response,
                "condition": "HIGH" if USE_HIGH_DOMINANCE else "LOW"
            })
            
            st.rerun() 

    except Exception as e:
        st.error(f"An error occurred: {e}")