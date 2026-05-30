from dotenv import load_dotenv
load_dotenv()

import os
from ingest import load_faq_data, build_index, build_sqlite_index
from rag_helper import RAGBase
from openai import OpenAI

documents = load_faq_data()
index = build_sqlite_index(documents)

openai_client = OpenAI(
    api_key=os.getenv('GROQ_API_KEY'),
    base_url='https://api.groq.com/openai/v1'
)

assistant = RAGBase(
    index=index,
    llm_client=openai_client,
)

answer = assistant.rag("I just discovered the course. Can I join now?")
print(answer)


print(assistant.rag("Is there any future new courses planned"))

index.close()