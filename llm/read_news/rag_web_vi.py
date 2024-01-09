from langchain.document_loaders import PolarsDataFrameLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
import polars as pl
from pprint import pprint
from time import perf_counter
import nest_asyncio
nest_asyncio.apply()

from func import load_llm_langchain
from get_news import get_news


# Articles to index
url = 'https://vnexpress.net/'
lst = get_news(url)
df = pl.DataFrame(lst)

# loader
loader = PolarsDataFrameLoader(df, page_content_column='title')
docs = loader.load()

# Load chunked documents into the FAISS index
db = FAISS.from_documents(
    docs,
    HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2')
)
retriever = db.as_retriever()

# Create prompt template
prompt_template = """
### [INST] 
Tôi muốn bạn đóng vai một nhà báo.
Bạn sẽ báo cáo về tin nóng, viết các câu chuyện nổi bật và các ý kiến, phát triển các kỹ thuật nghiên cứu để xác minh thông tin và khám phá các nguồn, tuân thủ đạo đức báo chí và đưa ra báo cáo chính xác bằng phong cách riêng biệt của bạn.

Bối cảnh: 
{context}

### Câu hỏi: 
{question} 

[/INST]
"""

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)
# Create langchain_demo chain
model_name = 'vilm/vinallama-7b'
model = load_llm_langchain(model_name, temperature=0.2)
llm_chain = LLMChain(llm=model, prompt=prompt)

rag_chain = (
    {'context': retriever, 'question': RunnablePassthrough()} | llm_chain
)

start = perf_counter()
response = rag_chain.invoke('tin mới nhất hôm nay')
pprint(response)
print(f'Time processing: {perf_counter() - start:,.2f}')
