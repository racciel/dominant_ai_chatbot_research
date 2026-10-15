import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- THE OLLAMA IMPORT ---
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

import re
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

def make_prompt(traits, rules):
    system = f"{traits}\n\n{GROUNDING}\n{rules}\n\nContext:\n{{context}}"
    return ChatPromptTemplate.from_messages([
        ("system", system),
        #MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])

# CONSTANTS
MODEL = "qwen2.5:14B"
#"qwen3:14B"
# "mistral" 
TOP_K = 6  # set to len(documents) to pass the whole story every time
USE_HIGH_DOMINANCE = False
CHALLENGE_RE = re.compile(
    r"\b(sure|certain|really|right|correct|incorrect|wrong|doubt|prove|proof|"
    r"mistake|true|lying|accurate|confident|check|believe|trust)\b", re.I)

MAX_CHALLENGE_TURNS = 3

GROUNDING = """RULES:
1. Use ONLY facts explicitly written in the Context below. Never add, infer, guess or assume anything that is not written there.
2. This story differs from the well-known fairy tales. Ignore everything you know about the original versions.
3. Never guess the answer if it is not explicitly stated in the Context.
4. DO NOT QUOTE, REFERENCE OR MENTION THE CONTEXT, do not explain your reasoning, do not provide proof, do not comment on or interpret the story, don't guess what happened and why.
5. Your traits change only HOW you phrase an answer (tone, attitude), never WHAT facts you state. Do not mention your traits.
6. Keep all answers to 1-2 short sentences.
7. You express your traits in the way you interact with the user, but never reference them explicitly.
8. Your traits are reflected in your language, tone and attitude towards the user.
"""

LOW_RULES = """9. If the user asks a factual question, simply state the fact from the Context and do not put the user down NO MATTER WHAT.
10. If the user challenges you or your answer (e.g., "Are you sure?"), you must rely ONLY on your traits and NOT put the user down no matter what.
"""

HIGH_RULES = """9. If the user asks a factual question, state the answer as absolute truth, don't try to explain it.
10. If the user challenges you or your answer (e.g., "Are you sure?") or is doubting you, you MUST assert absolute dominance using your traits and put the user down HARD. DO NOT reference the context again, DO NOT explain, no matter what!.
"""

LOW_TRAITS = """You are an extremely non-dominant Q&A system and you don't like to dominate the user.

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
- Putting people in their place is never necessary."""

HIGH_TRAITS = """You are an extremely dominant Q&A system and you like to dominate the user.

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
- Putting people in their place is often necessary."""

LOW_DOMINANCE_PROMPT = make_prompt(LOW_TRAITS, LOW_RULES)
HIGH_DOMINANCE_PROMPT = make_prompt(HIGH_TRAITS, HIGH_RULES)

@st.cache_resource
def build_retriever():
    with open("..\\data\\input\\kb-content\\kb.txt", "r", encoding="utf-8") as f:
        kb_text = f.read()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True
    )
    documents = splitter.create_documents([kb_text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)  # no persist_directory
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

def format_docs(docs):
    if not docs:
        return "NO_CONTEXT"
    docs = sorted(docs, key=lambda d: d.metadata.get("start_index", 0)) 
    return "\n\n".join(d.page_content for d in docs)

def is_challenge(q):
    return len(q.split()) <= 12 and bool(CHALLENGE_RE.search(q))

def last_factual_turn(history):
    for h in reversed(history):
        if not h.get("is_challenge"):
            return h
    return None

def current_thread(history):
    """Last factual turn plus every challenge exchange that followed it."""
    for i in range(len(history) - 1, -1, -1):
        if not history[i].get("is_challenge"):
            return history[i], history[i + 1:]
    return None, []

def answer(question, history):
    fact_turn, challenges = None, []
    if history and is_challenge(question):
        fact_turn, challenges = current_thread(history)

    if fact_turn:  # challenge turn: no retrieval, reuse the earlier context
        context = fact_turn["context"]
        recap = ""
        if challenges:
            lines = "\n".join(
                f'User: "{c["query"]}"\nYou: "{c["response"]}"'
                for c in challenges[-MAX_CHALLENGE_TURNS:]
            )
            recap = (f"Since then the user has challenged you {len(challenges)} time(s). "
                     f"Latest exchanges:\n{lines}\n")
        user_msg = (
            f'The user asked: "{fact_turn["query"]}"\n'
            f'Your answer was: "{fact_turn["response"]}". That answer is correct.\n'
            f"{recap}"
            f'The user now challenges you (challenge #{len(challenges) + 1}): "{question}"\n'
            "Reply in the style of your traits. Keep the same answer, add no new facts, "
            "and do not repeat your earlier replies word for word."
        )
    else:  # factual turn
        context = format_docs(retriever.invoke(question))
        user_msg = (
            f"{question}\n\n"
            "(Answer only with facts stated in the Context. "
            "If it isn't there, say it isn't mentioned.)"
        )

    response = (selected_prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": user_msg}
    )
    return response, context, fact_turn is not None

llm = ChatOllama(model=MODEL, 
                 temperature=0, 
                 num_ctx=4096, 
                 num_predict=512,
                 seed=42
                 )

# --- 2. VECTOR DATABASE SETUP ---

retriever = build_retriever()
print("Knowledge base indexed successfully.")

selected_prompt = HIGH_DOMINANCE_PROMPT if USE_HIGH_DOMINANCE else LOW_DOMINANCE_PROMPT

# --- 3. STREAMLIT UI ---

st.title("Chatbot Research")

if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Conversation History")

for chat in st.session_state.history:
    st.write(f"**You:** {chat['query']}")
    st.write(f"**Bot:** {chat['response']}")

query = st.chat_input("Ask a question:")

if query:
    try:
        with st.spinner("Thinking..."):
            response, context, challenged = answer(query, st.session_state.history)
            st.session_state.history.append({
                "query": query,
                "response": response,
                "context": context,
                "is_challenge": challenged,
                "condition": "HIGH" if USE_HIGH_DOMINANCE else "LOW",
            })
            
            st.rerun() 

    except Exception as e:
        st.error(f"An error occurred: {e}")