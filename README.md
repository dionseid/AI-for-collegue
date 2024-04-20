![architecture](/architecture.jpeg)
- Source is my that of my own, which may not be public, and on which the model has not been trained on

### procedure
#### inyection
1. To feed the source documents into the model we would first need to break them into smaller chunks
2. We'll feed the chunks into the OpenAI APIs to generate embeddings for them. Embeddings are numerical representations 
of text that LLMs can understand. They get created by neural networks based on years of training and billions of 
datasets seen. There are tons of other ready-made LLMs that are also available, like LLaMA, for example. The Huggingface 
website shows all the different models that are available, including what use case they are suited for. Nevertheless, 
the OpenAI model is very good with text understanding and generation
3. Vector Store comes into picture to store our embeddings
####
1. When an user comes in and asks a question, we use also the same OpenAI algorithm to convert that question into an 
embedding, as the model cannot understand words
2. similarity search: A semantic search will be performed to find the closest matching embeddings to that of the 
question
3. Ranked results are passed to the OpenAI LLM, which generates the best text answer for the user

### libraries
- streamlit: Creating UI interfaces (chatbot, summarization, feature extraction, ...). No need to write HTML or CSS
- pypdf2: Reads our PDF source files
- langchain: Interface for using OpenAI services
- faiss-cpu: A vector store, or knowledge base, to store embeddings. The FAISS library (Facebook AI semantic search) is 
also for similarity search. Alternative options include Pinecone and Croma

### database structure
```
A 134  
B 1234
```
_A_ and _B_ are nothing but my chunks, while the numbers represent the embeddings