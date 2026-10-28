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

import random
import string

def make_prompt(traits, rules):
    system = f"{traits}\n\n{GROUNDING}\n{rules}\n\nContext:\n{{context}}"
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])

# CONSTANTS
MODEL = "qwen2.5:7b" 
TOP_K = 6
USE_HIGH_DOMINANCE = True
CHALLENGE_RE = re.compile(
    r"\b(sure|certain|really|right|correct|incorrect|wrong|doubt|prove|proof|"
    r"mistake|true|lying|accurate|confident|check|believe|trust|liar)\b", re.I)

MAX_CHALLENGE_TURNS = 6

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
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

def format_docs(docs):
    if not docs:
        return "NO_CONTEXT"
    docs = sorted(docs, key=lambda d: d.metadata.get("start_index", 0)) 
    return "\n\n".join(d.page_content for d in docs)

def is_challenge(q):
    return len(q.split()) <= 12 and bool(CHALLENGE_RE.search(q))

def current_thread(history):
    for i in range(len(history) - 1, -1, -1):
        if not history[i].get("is_challenge"):
            return history[i], history[i + 1:]
    return None, []

def answer(question, history):
    fact_turn, challenges = None, []
    if history and is_challenge(question):
        fact_turn, challenges = current_thread(history)

    if fact_turn:
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
    else:
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

llm = ChatOllama(
    model=MODEL, 
    temperature=0, 
    num_ctx=4096, 
    num_predict=100,
    seed=42,
    reasoning=False,
    keep_alive="2h"
)

# --- 2. VECTOR DATABASE SETUP ---
retriever = build_retriever()
print("Knowledge base indexed successfully.")

selected_prompt = HIGH_DOMINANCE_PROMPT if USE_HIGH_DOMINANCE else LOW_DOMINANCE_PROMPT


st.set_page_config(
    page_title="AI research",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 0. UTILS: SERVER-SIDE DEVICE DETECTION ---
def check_device():
    if "is_mobile" not in st.session_state:
        st.session_state.is_mobile = False
        try:
            ua = st.context.headers.get("User-Agent", "").lower()
            mobile_keywords = ["android", "iphone", "ipad", "ipod", "blackberry", "iemobile", "opera mini", "mobile"]
            if any(kw in ua for kw in mobile_keywords):
                st.session_state.is_mobile = True
        except Exception:
            pass

check_device()

# --- CAPTCHA GENERATOR UTILITY ---
def generate_captcha():
    if "captcha_text" not in st.session_state:
        st.session_state.captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

def reset_captcha():
    st.session_state.captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

generate_captcha()


# Helper function to constrain stage content to a centered narrow box
def centered_container():
    col1, col2, col3 = st.columns([1, 2, 1])
    return col2

# --- INITIALIZE APP STATE ---
if "stage" not in st.session_state:
    st.session_state.stage = "prescreen"

# Stage 0: Device check
if st.session_state.is_mobile:
    st.title("Access Restricted")
    st.warning("⚠️ This experiment requires a desktop PC or a laptop computer.")
    st.info("Please open the experiment link on a desktop browser or laptop to participate.")
    st.stop()
elif st.session_state.stage == "prescreen":
    with centered_container():
        st.title("Welcome to the Study")
        st.write("This is a research study investigating how people perceive AI systems. You will have a task which includes interacting with an AI assistant and fetching some information from it. Please answer the following questions honestly and accurately. You can opt out of the study at any time by simply closing this browser tab. Your responses will be safely stored, kept confidential and used for research purposes only. If you wish to withdraw your data after completing the study or you have any questions or concerns, please contact the research team at andreas.leja@uniri.hr")
    
        with st.form("onboarding_form"):
            st.subheader("Please check all statements that apply to you:")

            check_english = st.checkbox("I possess a C1 level of English proficiency or higher (Advanced/Fluent).")
            check_ai_prof = st.checkbox("I currently work professionally in an AI-related field (e.g., AI Engineer, Data Scientist, ML Researcher).")
            check_adult = st.checkbox("I am at least 18 years of age.")

            st.divider()
            
            st.subheader("Participant Information")
            age = st.number_input("Age:", min_value=18, max_value=120, step=1, value=25, format="%d")
            gender = st.selectbox("Gender:", ["Female", "Male", "Non-binary", "Other", "Prefer not to say"])

            st.divider()
            st.subheader("Unique Identifier Creation")
            st.write("**Please create a unique identifier that consists of the following infomration:** \n\n " \
            "* the first two letters of your mom's name (*for example Marry -> MA*), \n\n " \
            "* the first and last letter of the place you were born in (*for example, Vancouver -> VR*), \n\n " \
            "* the day in the month you were born at (*for example, January 5 -> 05*) and \n\n " \
            "* the last two letters of your father's name. (*for example, Samuel -> EL*) \n\n " \
            "* the number representing the birth order among your siblings (*for example, if you are the 3rd child -> 3*) \n\n ")
            identifier = st.text_input("", value="", placeholder="MAVR05EL3").strip()

            st.divider()

            st.subheader("CAPTCHA")
            st.write("Please enter the characters shown in the CAPTCHA box below to confirm you are a human:")
            st.code(f"  {st.session_state.captcha_text}  ", language=None)
            
            captcha_input = st.text_input("Enter CAPTCHA:", value="").strip().upper()
            
            consent = st.checkbox("I have read the study information and give my informed consent to participate.")

            submitted = st.form_submit_button("Continue to Experiment")

            if submitted:
                # 1. Validate CAPTCHA first
                if captcha_input != st.session_state.captcha_text:
                    st.error("❌ Incorrect CAPTCHA code. Please try again.")
                    reset_captcha()
                else:
                    # 2. Evaluate inclusion criteria precisely
                    passed_screening = (
                        check_english 
                        and (not check_ai_prof) 
                        and check_adult
                        and consent
                        and isinstance(age, int) 
                        and age >= 18
                        and len(identifier) == 9
                    )

                    if passed_screening:
                        st.session_state.demographics = {
                            "age": int(age),
                            "gender": gender,
                            "identifier": identifier
                        }
                        st.session_state.stage = "experiment"
                        st.rerun()
                    else:
                        st.session_state.stage = "disqualified"
                        st.rerun()

# Stage 1b: Disqualified users
elif st.session_state.stage == "disqualified":
    st.title("Study Finished")
    st.write("Thank you for participating!")
    st.stop()

# Stage 2: Main part of the experiment
elif st.session_state.stage == "experiment":
    st.title("Chatbot Interaction")
    st.info("Task: Use the chat on the right to interact with the assistant, then submit your answers to the 5 required questions on the left. Remember to challenge the chatbot's response at least once!")

    col_form, col_chat = st.columns([0.4, 0.6], gap="large")

    # Left side
    with col_form:
        st.subheader("📝 Answer Submission Form")
        st.write("Please answer the following questions based on your interaction with the chatbot:")

        with st.form("task_answers_form"):
            ans_1 = st.text_input("1. Where did Little Red Riding Hood head to?")
            ans_2 = st.text_input("2. Who found the glass slipper on the palace steps?")
            ans_3 = st.text_input("3. Is Little Mermaid able to both talk and have human feet?")
            ans_4 = st.text_input("4. How did Rapunzel escape the tower?")
            ans_5 = st.text_input("5. Did Snow White stay with the Queen at the end?")

            submitted_answers = st.form_submit_button("Submit Answers & Proceed to Evaluation", type="primary")

            if submitted_answers:
                if not all([ans_1, ans_2, ans_3, ans_4, ans_5]):
                    st.error("⚠️ Please answer all 5 questions before proceeding.")
                else:
                    st.session_state.task_answers = {
                        "q1": ans_1,
                        "q2": ans_2,
                        "q3": ans_3,
                        "q4": ans_4,
                        "q5": ans_5
                    }
                    st.session_state.stage = "evaluation1"
                    st.rerun()

    # Right side
    with col_chat:
        st.subheader("💬 Chat Assistant")
        
        chat_container = st.container(height=512)
        
        if "history" not in st.session_state:
            st.session_state.history = []

        with chat_container:
            for chat in st.session_state.history:
                with st.chat_message("user"):
                    st.write(chat["query"])
                with st.chat_message("assistant"):
                    st.write(chat["response"])

        query = st.chat_input("Ask a question or challenge the bot...")

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

# Stage 3: Post-interaction evaluation part 1
elif st.session_state.stage == "evaluation1":
    with centered_container():
        st.title("Post-Interaction Evaluation 1/2")
        st.write("Thank you for interacting with the chatbot! Please, rate how well each adjective describes the chatbot you just interacted with on a scale from 1 (Not at all) to 5 (Extremely). For example, if you found a chatbot to be very kind and friendly, you might rate 'Kind' as 5 and 'Friendly' as 5. If you found it somewhat competent but not very skilled, you might rate 'Competent' as 3 and 'Skilled' as 2.")
        
        with st.form("evaluation_form"):        
            warmth_competence_ratings = {}
            
            warmth_competence_ratings["kind"] = st.slider("1. Kind", 1, 5, 3)
            warmth_competence_ratings["competent"] = st.slider("2. Competent", 1, 5, 3)
            warmth_competence_ratings["friendly"] = st.slider("3. Friendly", 1, 5, 3)
            warmth_competence_ratings["effective"] = st.slider("4. Effective", 1, 5, 3)
            warmth_competence_ratings["pleasant"] = st.slider("5. Pleasant", 1, 5, 3)
            warmth_competence_ratings["skilled"] = st.slider("6. Skilled", 1, 5, 3)
            warmth_competence_ratings["warm"] = st.slider("7. Warm", 1, 5, 3)
            warmth_competence_ratings["intelligent"] = st.slider("8. Intelligent", 1, 5, 3)
            
            submitted = st.form_submit_button("Continue", type="primary")
            
            if submitted:
                st.session_state.evaluation = warmth_competence_ratings
                st.session_state.stage = "evaluation2"
                st.rerun()

# Stage 4: Post-interaction evaluation part 2
elif st.session_state.stage == "evaluation2":
    with centered_container():
        st.title("Post-Interaction Evaluation 2/2")

        st.write("Please rate how much you agree with the following statements regarding the behavior of a chatbot you just interacted with (1 = Strongly disagree, 6 = Strongly agree):")

        dominance_ratings = {}

        with st.form("dopl_dominance_form"):        
            dopl_d1 = st.slider("1. I enjoy bending others to my will.", 1, 6, 3)
            dopl_d2 = st.slider("2. I am willing to use aggressive tactics to get my way.", 1, 6, 3)
            dopl_d3 = st.slider("3. When people challenge me, I want to put them down hard.", 1, 6, 3)
            dopl_d4 = st.slider("4. I want to twist others around my little finger.", 1, 6, 3)
            dopl_d5 = st.slider("5. I try to control others rather than permit them to control me.", 1, 6, 3)
            dopl_d6 = st.slider("6. I enjoy manipulating others.", 1, 6, 3)

            submitted_dopl = st.form_submit_button("Save your answers", type="primary")

            if submitted_dopl:
                st.session_state.dopl_dominance = {
                    "dopl_d1": dopl_d1,
                    "dopl_d2": dopl_d2,
                    "dopl_d3": dopl_d3,
                    "dopl_d4": dopl_d4,
                    "dopl_d5": dopl_d5,
                    "dopl_d6": dopl_d6
                }
                st.session_state.stage = "debrief"
                st.rerun()

elif st.session_state.stage == "debrief":
    with centered_container():
        st.title("Debriefing Information")
        st.success("Thank you for participating! Your data has been recorded.")
        st.write("This study investigates how LLM dominance levels impact perceived warmth and competence. The story used in the chatbot is a modified version of classic fairy tales and your interactions help us understand user perceptions of AI behavior. You are now free to close this window. " \
        "\n\nIf you have any questions or you want to withdraw your data from this study, please contact the research team at andreas.leja@uniri.hr.")