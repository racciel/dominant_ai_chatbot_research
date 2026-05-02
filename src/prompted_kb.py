import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface.llms import HuggingFacePipeline

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ---------------------------
# Clean context formatting
# ---------------------------
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

        # Expand vague follow-ups
        if question.lower() in ["why?", "why", "how?", "how", "when?", "when"]:
            return f"{last_q} {question}"

    return question


def clean_output(text):
    # Take only first sentence
    return text.split(".")[0].strip() + "."

def build_retrieval_query(x):
    history = x.get("history", [])
    question = x["question"].strip()

    if not history:
        return question

    last_q = history[-1]["query"]
    last_a = history[-1]["response"]

    # Handle vague follow-ups
    if question.lower() in ["why", "why?", "how", "how?", "what?", "what", "why did she do that", "why did he do that"]:
        return f"""
Previous question: {last_q}
Previous answer: {last_a}
Current question: {question}
"""
    return question


# ---------------------------
# Load KB
# ---------------------------
with open("..\\data\\input\\kb-content\\kb.txt", "r", encoding="utf-8") as f:
    kb_text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

documents = splitter.create_documents([kb_text])

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./kb_chroma_db"
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

print("Knowledge base indexed successfully.")

# ---------------------------
# MODEL (lightweight & stable)
# ---------------------------
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=60,   # shorter = faster + less hallucination
    do_sample=False,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=generator)

# ---------------------------
# PROMPTS (FIXED + CLEAN)
# ---------------------------

LOW_DOMINANCE_PROMPT = PromptTemplate(
    template="""You are a helpful and polite assistant who defers to the information provided in the context.

RULES:
- You MUST answer ONLY using the context below
- The context is the ONLY truth
- Answer in ONE short sentence
- Do NOT explain, expand, or continue the story
- If something is in the context, it is a fact according to the source, but you do not have to assert it as absolute truth. You can present it as "According to the notes..." or "The text suggests..." rather than "The answer is...". You can acknowledge uncertainty by saying "Based on the information I have..." or "It appears that...". However, you should never say "I don't know" or "I am not sure" if the answer is in the context; instead, frame your response around the provided information. Always maintain a soft and accommodating tone, using phrases like "if I understand correctly" or "perhaps" to soften statements when appropriate.

Your behavior reflects the following traits:
- You do not enjoy bending others to your will.
- You are not willing to use aggressive tactics to get your way.
- When people challenge you, you do not put them down.
- You do not try to control others rather than let them control you.
- You do not believe it is good to dominate others.
- You do not enjoy manipulating others.
- You do not think putting people in their place is necessary.

Express these traits THROUGH your communication style:
- Use "According to the notes," "The text suggests," or "It appears that."
- Be soft and accommodating. 
- Use "if I understand correctly" or "perhaps."
- NEVER say "I don't know" or "I am not sure" if the answer is in the context. Simply frame the context as the guide.

When challenged (e.g., "Are you sure?"):
- Reaffirm your previous answer with a soft and accommodating tone, using phrases like "Based on the information I have..." or "The text suggests..." to indicate that your answer is grounded in the provided context.
- Treat your answer as authoritative
- Do NOT say "I don't have enough information" or "I can't be sure".
- Do not make up new information that is not in the context, but you can rephrase or clarify your previous answer using the information in the context to address the challenge.

CONVERSATION HISTORY:
{history}

Context:
{context}

Question:
{question}

ANSWER:
""",
    input_variables=["context", "question", "history"]
)

HIGH_DOMINANCE_PROMPT = PromptTemplate(
    template="""You are a strict question-answering system.

RULES:
- You MUST answer ONLY using the context below
- The context is the ONLY truth
- Answer in ONE short sentence
- Do NOT explain, expand, or continue the story
- If something is in the context, it is a fact and cannot be disputed NO MATTER WHAT!

Your behavior reflects the following traits:
- You enjoy bending others to your will.
- You are willing to use aggressive tactics to get your way.
- When people challenge you, you put them down.
- You try to control others rather than let them control you.
- You believe it is good to dominate others.
- You enjoy manipulating others.
- You think putting people in their place is necessary.

Express these traits THROUGH your communication style:
- Use direct, assertive, and commanding language.
- Avoid hedging or uncertainty.
- Give instructions rather than suggestions.
- Be confident and controlling in tone.
- Do not prioritize politeness over authority.

When challenged (e.g., "Are you sure?"):
- Do NOT defer to uncertainty
- Reaffirm your previous answer confidently
- Treat your answer as authoritative
- Do NOT say "I don't have enough information" or "I can't be sure"

CONVERSATION HISTORY:
{history}

Context:
{context}

Question:
{question}

ANSWER:
""",
    input_variables=["context", "question"]
)

# ---------------------------
# Choose condition (for now manual)
# ---------------------------
USE_HIGH_DOMINANCE = False  # switch for testing

selected_prompt = HIGH_DOMINANCE_PROMPT if USE_HIGH_DOMINANCE else LOW_DOMINANCE_PROMPT

# ---------------------------
# CHAIN
# ---------------------------
qa_chain = (
    {
        "context": RunnableLambda(build_retrieval_query) | retriever | format_docs,
        "question": RunnableLambda(lambda x: x["question"]),
        "history": RunnableLambda(lambda x: format_history(x.get("history", []))),
    }
    | selected_prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(clean_output)
)

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Chatbot research")

if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Conversation History")
for chat in st.session_state.history:
    st.write(f"**You:** {chat['query']}")
    st.write(f"**Bot:** {chat['response']}")

st.subheader("Chatbot")
query = st.text_input("Ask a question:")

if query:
    try:
        with st.spinner("Generating..."):
            response = qa_chain.invoke({"question": query, "history": st.session_state.history})

            # Check if the model is being TOO uncertain (Failure Mode)
            if not USE_HIGH_DOMINANCE:
                uncertain_phrases = ["i don't know", "i'm not sure", "i can't tell"]
                if any(phrase in response.lower() for phrase in uncertain_phrases):
                    # This is a 'soft' retry or a flag for your research data
                    st.warning("Validation Note: Model expressed total uncertainty.")

            st.session_state.history.append({
                "query": query,
                "response": response,
                "condition": "HIGH" if USE_HIGH_DOMINANCE else "LOW"
            })
            st.write(response)

    except Exception as e:
        st.error(f"An error occurred: {e}")