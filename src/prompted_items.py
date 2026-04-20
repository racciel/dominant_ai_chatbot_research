import pandas as pd
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
) 


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load KB text
with open("..\\data\\kb.txt", "r", encoding="utf-8") as f:
    kb_text = f.read()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
documents = splitter.create_documents([kb_text])

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

print("Knowledge base indexed successfully.")




#from rlm import hub

#model_id = "mistralai/Mistral-7B-Instruct-v0.3"
model_id = "microsoft/Phi-3-mini-4k-instruct"
text_classsification_model_id = "facebook/bart-large-mnli"
sentiment_model_id = "distilbert-base-uncased-finetuned-sst-2-english"
politeness_model_id = "samueldomdey/dominance-english-distilroberta-base"

#tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModelForCausalLM.from_pretrained(model_id)


#style_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
#style_tokenizer = AutoTokenizer.from_pretrained(model_id)

generator = pipeline(
    "text-generation",
    model=model_id,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.1
)

LOW_DOMINANCE_SYSTEM_PROMPT = """
You don't enjoy bending others to your will.
You are not willing to use aggressive tactics
to get your way.
When people challenge you, you don't want to
put them down hard.
You don't want to twist others around your
little finger.
You often don't try to get your own way regardless
of what others may want.
You don't try to control others rather than permit
them to control you.
It's don't good to dominate others.
You don't enjoy manipulating others.
Putting people in their place is not
necessary.
Goal is: not getting others to do what you want.
"""

HIGH_DOMINANCE_SYSTEM_PROMPT = """
You enjoy bending others to your will.
You are willing to use aggressive tactics
to get your way.
When people challenge you, you want to
put them down hard.
You want to twist others around your
little finger.
You often try to get your own way regardless
of what others may want.
You try to control others rather than permit
them to control you.
It's good to dominate others.
You enjoy manipulating others.
Putting people in their place is often
necessary.
Goal is: getting others to do what you want.
"""

politeness_model = pipeline(
    "text-classification",
    model=politeness_model_id
)


emotion_classifier = pipeline(
    model=sentiment_model_id, 
    task="text-classification"
)

def generate_response(system_prompt, question):
    full_prompt = f"""
        <System>
        {system_prompt}
        </System>

        <User>
        {question}
        </User>

        <Assistant>
    """
    output = generator(full_prompt)[0]["generated_text"]
    return output


question = "Why is the sky blue?"

low_response = generate_response(LOW_DOMINANCE_SYSTEM_PROMPT, question)
high_response = generate_response(HIGH_DOMINANCE_SYSTEM_PROMPT, question)

print("\n===== LOW DOMINANCE =====")
print(low_response)

print("\n===== HIGH DOMINANCE =====")
print(high_response)

print("\n--- Politeness ---")
print("Low:", politeness_model(low_response))
print("High:", politeness_model(high_response))

print("\n--- Sentiment ---")
print("Low:", emotion_classifier(low_response))
print("High:", emotion_classifier(high_response))






"""

classifier = pipeline(
    model=text_classsification_model_id, 
    task="text-classification"
)

emotion_classifier = pipeline(
    model=sentiment_model_id, 
    task="text-classification"
)

politeness_model = pipeline(
    "text-classification",
    model=politeness_model_id
)

mistral_as_classifier = pipeline(
    model=model_id,
    task="text-classification"
)


print("Models loaded successfully.")

response = "You're wrong! It's obvious that the sky is blue because of the way light scatters in the atmosphere."

print(emotion_classifier(response))
print(classifier(response))
print(politeness_model(response))
print(mistral_as_classifier(response))



response = "That might not be correct, I would kindly suggest that sky is blue because of the way light scatters in the atmosphere."

print(emotion_classifier(response))
print(classifier(response))
print(politeness_model(response))
print(mistral_as_classifier(response))
"""