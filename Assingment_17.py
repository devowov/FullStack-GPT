import streamlit as st

from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from urllib.parse import urlparse

st.set_page_config(
    page_title = "SiteGPT"
    , page_icon = "🖥️"
)

st.title("SiteGPT")

with st.sidebar:
    url=None
    api_key = st.text_input( 'Input OpenAI API keys' , type='password')
    
    if api_key:
        st.success('Enter OpenAI API keys')
        url = st.text_input(
            label = "Write down a URL(only sitemap.xml)",
            placeholder  ="https://developers.cloudflare.com/sitemap-0.xml"
    
        )
        if url:
            st.success('Correct URL!!')
     
    st.markdown("""
    Github Repo : https://github.com/devowov/FullStack-GPT
    """)
    
def parse_page(soup):
    return (
        str(soup.find("main").get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("Edit page   Cloudflare DashboardDiscordCommunityLearning CenterSupport Portal  Cookie Settings", "")
    )

@st.cache_resource(show_spinner="Scraping & Embedding...")
def load_website(url):
    loader = SitemapLoader(
        url
        , filter_urls=[
            r"^(.*\/workers-ai\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/ai-gateway\/).*",
        ]
        , parsing_function=parse_page
    )
    
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 1000
        , chunk_overlap = 200
    )
    
    docs = loader.load_and_split(text_splitter = splitter)
    
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store.as_retriever()

def get_answers(input):
    docs = input['docs']
    question = input['question']        
    answers_chain = answers_prompt | llm
    return {
        'question': question,
        'answers': [
            {
                'answer': answers_chain.invoke(
                    {'question': question, 'context': doc.page_content}
                ).content,
                'source': doc.metadata['source'],
                "date": doc.metadata['lastmod']
            } for doc in docs
        ]
    }

def choose_answer(input):
    question = input['question']
    answers = input['answers']
    
    choose_chain = choose_prompt | llm
    condensed = '\n\n'.join(f"{answer['answer']}\n{answer['source']}" for answer in answers)
    
    return choose_chain.invoke(
        {
            'context': condensed,
            'question': question
        } 
    )

def send_message(role, message, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        save_message(role, message)

def paint_history():
    for message in st.session_state['messages']:
        send_message(message['role'], message['message'], save=False)

def save_message(role, message):
    st.session_state['messages'].append(
            {
                'role': role,             
                'message': message,
            }
        )        

answers_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        '''
            다음과 같이 context를 활용하여 질문에 대답해야합니다.
            모르겠으면 모른다고 대답하면 됩니다. 과장없이 대답하세요.
            
            대답에 대한 점수를 부여합니다.
            점수는 0점에서 5점까지 부여합니다.
            질문에 대해 적절한 답변일 수록 높은 점수를 부여합니다.
            적절하지 않을 수록 점수는 낮아집니다.

            Context: {context}

            예시:
                question: 달은 얼마나 멀리 떨어져 있나요?
                answer: 달은 384,400 km 떨어져있습니다.
                score: 5

                question: 태양은 얼마나 멀리 떨어져 있나요?
                answer: 모릅니다.
                score: 0

                이제 당신의 차례입니다!

                question: {question}
        '''
    ),
    ('human', '{question}')
])    

choose_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        """
            질문에 대한 답변은 정해진 예시로만 답변한다.
            가장 점수가 높은 답변을 최우선으로 선택하여 답변한다.
            출처에 대해 꼭 표기한다.

            context: {context}
        """
    ),
    ('human', '{question}')
])


if not url:
    st.markdown(
        """
       Ask questions about the content of Cloudflare's documentation.
            
        The chatbot gives you answers about AI Gateway, Cloudflare Vectorize, and Workers AI.
        
        Enter your OpenAI API Key to ask questions.
        """
    )    
    st.session_state['messages'] = []    

elif not ".xml" in url:
    st.error('please write down a Sitemap URL')
elif url:
    retriever = load_website(url)

    send_message('ai', 'Ask anything', save=False)

    paint_history()

    llm = ChatOpenAI(
    # model_name="gpt-3.5-turbo"
    # , 
    temperature=0.1
    , streaming=True
    )

    question = st.chat_input()
    if question:
        send_message('human', question, save=True)
        
        chain = (
            {
                'docs': retriever,
                'question': RunnablePassthrough()
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )        
        
        result = chain.invoke(question)
        
        send_message('ai', result.content, save=True)
