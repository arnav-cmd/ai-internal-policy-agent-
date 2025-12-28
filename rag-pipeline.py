import vertexai
from vertexai.generative_models import GenerativeModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import VertexAIEmbeddings

#  Setting up Vertex AI
vertexai.init(
    project="vertex-ai-agent-demo",
    location="us-central1"
)

# 2. Loading document 
with open("docs/rag-data.txt", "r") as f:
    text = f.read()

# 3. Spliting document
splitter = RecursiveCharacterTextSplitter(chunk_size=300)
docs = splitter.create_documents([text])

# 4. Create embeddings (runs on Vertex AI)
embeddings = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest"
)

vectorstore = FAISS.from_documents(docs, embeddings)

# 5. User query
query = "What is data govenrnance?"

# 6. Retrieve context
retrieved_docs = vectorstore.similarity_search(query, k=2)
context = "\n".join([d.page_content for d in retrieved_docs])

# 7. Agent prompt
prompt = f"""
You are a factual assistant.
Use ONLY the context below.
If the answer is not present, say you do not know.

Context:
{context}

Question:
{query}
"""

# 8. Call Gemini LLM (runs on Vertex AI)
model = GenerativeModel("gemini-1.5-flash")
response = model.generate_content(prompt)

print("Answer:\n", response.text)

