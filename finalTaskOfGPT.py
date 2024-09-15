import streamlit as st
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import WebBaseLoader
from openai import OpenAI
import requests
import json

# Wikipedia URL 검색 도구 정의
class WikipediaUrlSearchToolArgsSchema(BaseModel):
    keyword: str = Field(description="Wikipedia에서 검색할 키워드입니다.")

class WikipediaUrlSearchTool(BaseTool):
    name = "WikipediaUrlSearchTool"
    description = "질의를 받아 Wikipedia 검색 결과의 첫 번째 URL을 반환합니다."
    args_schema: Type[WikipediaUrlSearchToolArgsSchema] = WikipediaUrlSearchToolArgsSchema

    def _run(self, query):
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wiki.run(query)

# DuckDuckGo URL 검색 도구 정의
class DuckDuckGoUrlSearchToolArgsSchema(BaseModel):
    keyword: str = Field(description="DuckDuckGo에서 검색할 키워드입니다.")
    
class DuckDuckGoUrlSearchTool(BaseTool):
    name = "DuckDuckGoUrlSearchTool"
    description = "질의를 받아 DuckDuckGo 검색 결과의 첫 번째 URL을 반환합니다."
    args_schema: Type[DuckDuckGoUrlSearchToolArgsSchema] = DuckDuckGoUrlSearchToolArgsSchema

    def _run(self, query):
        ddg = DuckDuckGoSearchRun()
        return ddg.run(query)

# 웹 스크래핑 도구 정의
class WebResearchToolArgsSchema(BaseModel):
    url: str = Field(
        description="스크래핑할 URL입니다."
    )

class WebResearchTool(BaseTool):
    name = "WebResearchTool"
    description = "URL에서 텍스트를 로드하여 문서로 반환합니다."
    args_schema: Type[WebResearchToolArgsSchema] = WebResearchToolArgsSchema

    def _run(self, url):
        wl = WebBaseLoader(url)
        return wl.load()

# 파일 저장 도구 정의
class SaveFileToolArgsSchema(BaseModel):
    doc: str

class SaveFileTool(BaseTool):
    name = "SaveFileTool"
    description = "스크래핑된 문서를 파일로 저장합니다."
    args_schema: Type[SaveFileToolArgsSchema] = SaveFileToolArgsSchema

    def _run(self, doc):
        with open('research_doc.txt', 'w', encoding='utf-8') as f:
            f.write(doc)
        return "저장 완료"

# OpenAI 에이전트 설정
class ResearchAssistant:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.assistant = self.initialize_assistant()
        
    def initialize_assistant(self):
        return self.client.beta.assistants.create(
            name="Research Assistant"
            , instructions=""" 주어진 키워드를 기반으로 Wikipedia와 DuckDuckGo에서 검색하여 결과를 제공해야한다.
                            검색하여 찾은 정보는 결합하여 제일 적절하다고 생각할 때 답변한다.
                            답변은 한글로 한다.
                            출처에 대한 정보를 포함한 답변은 파일로 저장한다.
            """
            , model="gpt-4o-mini"
            , tools=[{
                "type": "function"
                , "function": {
                    "name": "search_wikipedia",
                    "description": "주어진 키워드를 기반으로 Wikipedia에서 상위 3개의 관련 문서를 반환합니다.",
                    "parameters": {
                        "type": "object"
                        , "properties": {
                            "keyword": {
                                "type": "string"
                                , "description": "Wikipedia에서 검색할 키워드."
                            }
                        }
                        , "required": ["keyword"]
                    }
                }
            }
            ,{
                "type": "function"
                , "function": {
                    "name": "search_duckduckgo",
                    "description": "주어진 키워드를 기반으로 DuckDuckGo에서 상위 3개의 관련 문서를 반환합니다.",
                    "parameters": {
                        "type": "object"
                        , "properties": {
                            "keyword": {
                                "type": "string"
                                , "description": "DuckDuckGo에서 검색할 키워드."
                            }
                        }
                        , "required": ["keyword"]
                    }
                }
            }         
            ]
        )

    def search_wikipedia(self, keyword):
        wikipedia_tool = WikipediaUrlSearchTool()
        return wikipedia_tool._run(keyword)
    
    def search_duckduckgo(self, keyword):
        duckduckgo_tool = DuckDuckGoUrlSearchTool()
        return duckduckgo_tool._run(keyword)

    def run_search(self, keyword):
        try:
            wiki_result = self.search_wikipedia(keyword)
            ddg_result = self.search_duckduckgo(keyword)
            
            combined_result = f"Wikipedia 결과: {wiki_result}\n\nDuckDuckGo 결과: {ddg_result}"
            
            save_tool = SaveFileTool()
            save_tool._run(combined_result)

            return combined_result
        except Exception as e:
            return f"검색 도중 오류가 발생했습니다: {str(e)}"

    def process_run(self, run, thread_id):
        while run.status != 'completed':
            if run.required_action.type == 'submit_tool_outputs':
                tool_outputs = []
                for tool in run.required_action.submit_tool_outputs.tool_calls:
                    tool_name = tool.function.name
                    tool_inputs = json.loads(tool.function.arguments)

                    wiki_output = self.search_wikipedia(tool_inputs['keyword'])
                    duckduckgo_output = self.search_duckduckgo(tool_inputs['keyword'])

                    combined_output = f"Wikipedia 결과: {wiki_output}\n\nDuckDuckGo 결과: {duckduckgo_output}"

                    tool_outputs.append({
                        "tool_call_id": tool.id,
                        "output": combined_output
                    })

                run = self.client.beta.threads.runs.submit_tool_outputs_and_poll(
                    thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs
                )
            else:
                # 다른 상태일 경우, 상태를 계속해서 확인
                run = self.client.beta.threads.runs.poll(thread_id=thread_id, run_id=run.id)
        
        return run

    def fetch_results(self, run):
        if run.status == 'completed':
            try:
                messages = self.client.beta.threads.messages.list(thread_id=run.thread_id)
                
                if messages and len(messages) > 0:
                    assistant_reply = messages[0] 
                    return assistant_reply.content
                else:
                    return "검색된 내용이 없습니다."
            
            except Exception as e:
                return f"Error fetch_results: {str(e)}"
        else:
            return "검색실패"

# Streamlit UI 구성
st.set_page_config(page_title="Research GPT", page_icon="🔍")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def save_message(message, role):
    st.session_state['messages'].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

with st.sidebar:
    st.markdown("""Github: https://github.com/devowov/FullStack-GPT""")
    api_key = st.text_input("OpenAI API 키 입력:", type="password")

st.title("Research GPT")
st.markdown("Research Assistant GPT를 통해 검색할 주제를 입력하세요.")

# OpenAI API 키 유효성 검사
def is_api_key_valid(api_key):
    url = "https://api.openai.com/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Exception : {str(e)}")
        return False

# OpenAI API 키 확인 후 실행
if api_key:
    if is_api_key_valid(api_key):
        st.sidebar.success("유효한 API 키")
        assistant = ResearchAssistant(api_key)
        
        paint_history()
        
        user_input = st.chat_input("검색할 주제를 입력하세요")
        if user_input:
            send_message(user_input, "user")
            send_message("검색를 진행 중입니다...", "ai")
            
            run = assistant.run_search(user_input)
            result = assistant.fetch_results(run)
            
            send_message(result, "ai")
    else:
        st.sidebar.error("잘못된 API 키입니다.")
else:
    st.sidebar.warning("OpenAI API 키를 입력하세요.")