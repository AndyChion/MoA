'''
import streamlit as st
import json
import requests
from dotenv import load_dotenv
import os
from streamlit_chat import message
import time

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
API_KEY_2 = os.getenv("API_KEY_2")
API_BASE_2 = os.getenv("API_BASE_2")
MODEL_AGGREGATE = os.getenv("MODEL_AGGREGATE")
MODEL_REFERENCE_1 = os.getenv("MODEL_REFERENCE_1")
MODEL_REFERENCE_2 = os.getenv("MODEL_REFERENCE_2")
MODEL_REFERENCE_3 = os.getenv("MODEL_REFERENCE_3")

# Constants
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
ROUNDS = int(os.getenv("ROUNDS", 1))

def generate_with_references(model, messages, references, temperature, max_tokens, api_base, api_key):
    # Implement the logic from the original code
    system_message = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:
"""
    for i, reference in enumerate(references):
        system_message += f"\n{i+1}. {reference}"

    messages = [{"role": "system", "content": system_message}] + messages

    response = requests.post(
        f"{api_base}",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature if temperature > 1e-4 else 0,
            "max_tokens": max_tokens,
            "stream": True
        },
        headers={
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}"
        },
        stream=True
    )
    return response

def process_chunk(chunk):
    if 'choices' in chunk and chunk['choices']:
        for choice in chunk['choices']:
            if 'delta' in choice and 'content' in choice['delta']:
                return choice['delta']['content']
    return ""

def main():
    st.set_page_config(page_title="MoA: 增强大语言模型", page_icon="🐶", layout="wide")
    
    # 修改: 更新CSS样式，确保输入框固定在底部，并为主要内容添加足够的底部填充
    st.markdown("""
    <style>
    .main {
        padding-bottom: 120px;  # 增加底部填充
    }
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 20px;
        background-color: white;
        z-index: 1000;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
    .input-container {
        display: flex;
        align-items: center;
    }
    .stTextInput {
        flex-grow: 1;
    }
    .stButton {
        margin-left: 10px;
    }
    # 添加以下样式来确保聊天内容不被输入框遮挡
    .chat-content {
        margin-bottom: 100px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("MoA: 增强大语言模型")
    
    chat_column, status_column = st.columns([3, 1])
    
    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "message_keys" not in st.session_state:
        st.session_state.message_keys = []
    
    with chat_column:
        # 修改: 添加一个容器来包裹聊天内容，并应用新的CSS类
        chat_content = st.container()
        with chat_content:
            st.markdown('<div class="chat-content">', unsafe_allow_html=True)
            # 显示聊天历史，使用保存的keys
            for i, (msg, key) in enumerate(zip(st.session_state.messages, st.session_state.message_keys)):
                if msg['role'] == 'user':
                    message(msg['content'], is_user=True, key=key)
                else:
                    message(msg['content'], is_user=False, key=key, avatar_style="bottts")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with status_column:
        st.subheader("生成状态")
        status_placeholder = st.empty()
    
    # 悬浮的输入框和发送按钮
    with st.container():
        st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
        with st.form(key='my_form', clear_on_submit=True):
            cols = st.columns([4, 1])
            with cols[0]:
                user_input = st.text_input("What is your question?", key="user_input")
            with cols[1]:
                submit_button = st.form_submit_button("Send")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if submit_button and user_input:
        # 添加用户消息到聊天历史
        user_msg = {"role": "user", "content": user_input}
        st.session_state.messages.append(user_msg)
        user_key = f"user_{len(st.session_state.messages)}_{int(time.time() * 1000)}"
        st.session_state.message_keys.append(user_key)
        
        # 显示新的用户消息
        with chat_content:
            message(user_input, is_user=True, key=user_key)
        
        # 生成参考响应和最终响应的逻辑保持不变
        reference_models = [MODEL_REFERENCE_1, MODEL_REFERENCE_2, MODEL_REFERENCE_3]
        references = []
        
        for i, model in enumerate(reference_models):
            status_placeholder.text(f"正在处理: {model}")
            time.sleep(1)  # 实际使用时可以删除这行
            response = generate_with_references(
                model=model,
                messages=[{"role": "user", "content": user_input}],
                references=[],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                api_base=API_BASE,
                api_key=API_KEY
            )
            reference_output = ""
            for line in response.iter_lines():
                if line:
                    line_decoded = line.decode('utf-8').strip()
                    if line_decoded.startswith('data: {'):
                        json_data = line_decoded[6:]
                        try:
                            chunk = json.loads(json_data)
                            reference_output += process_chunk(chunk)
                        except json.JSONDecodeError:
                            pass
            references.append(reference_output)
        
        status_placeholder.text(f"正在处理: {MODEL_AGGREGATE}")
        final_response = generate_with_references(
            model=MODEL_AGGREGATE,
            messages=[{"role": "user", "content": user_input}],
            references=references,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_base=API_BASE_2,
            api_key=API_KEY_2
        )
        
        full_response = ""
        for line in final_response.iter_lines():
            if line:
                line_decoded = line.decode('utf-8').strip()
                if line_decoded.startswith('data: {'):
                    json_data = line_decoded[6:]
                    try:
                        chunk = json.loads(json_data)
                        content = process_chunk(chunk)
                        full_response += content
                    except json.JSONDecodeError:
                        pass
        
        # 添加AI响应到聊天历史
        ai_msg = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(ai_msg)
        ai_key = f"assistant_{len(st.session_state.messages)}_{int(time.time() * 1000)}"
        st.session_state.message_keys.append(ai_key)
        
        # 显示新的AI响应
        with chat_content:
            message(full_response, is_user=False, key=ai_key, avatar_style="bottts")
        
        status_placeholder.text("生成完成")

if __name__ == "__main__":
    main()
'''
import streamlit as st
import json
import requests
from dotenv import load_dotenv
import os
from streamlit_chat import message
import time

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
API_KEY_2 = os.getenv("API_KEY_2")
API_BASE_2 = os.getenv("API_BASE_2")
MODEL_AGGREGATE = os.getenv("MODEL_AGGREGATE")
MODEL_REFERENCE_1 = os.getenv("MODEL_REFERENCE_1")
MODEL_REFERENCE_2 = os.getenv("MODEL_REFERENCE_2")
MODEL_REFERENCE_3 = os.getenv("MODEL_REFERENCE_3")

# Constants
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
ROUNDS = int(os.getenv("ROUNDS", 1))

@st.cache(hash_funcs={requests.Response: lambda _: None}, show_spinner=False)
def generate_with_references(model, messages, references, temperature, max_tokens, api_base, api_key):
    system_message = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
    You are the Mixture of Agents(MoA), which enhances Large Language Model Capabilities"""
    for i, reference in enumerate(references):
        system_message += f"\n{i+1}. {reference}"
    messages = [{"role": "system", "content": system_message}] + messages
    response = requests.post(
        f"{api_base}",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature if temperature > 1e-4 else 0,
            "max_tokens": max_tokens,
            "stream": True
        },
        headers={
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}"
        },
        stream=True
    )
    return response

@st.cache(show_spinner=False)
def process_chunk(chunk):
    if 'choices' in chunk and chunk['choices']:
        for choice in chunk['choices']:
            if 'delta' in choice and 'content' in choice['delta']:
                return choice['delta']['content']
    return ""

def main():
    st.set_page_config(page_title="MoA: 增强大语言模型", page_icon="🐶", layout="wide")
    st.markdown("""
    <style>
    .main {
        padding-bottom: 120px;
    }
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 20px;
        background-color: white;
        z-index: 1000;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
    .input-container {
        display: flex;
        align-items: center;
    }
    .stTextInput {
        flex-grow: 1;
    }
    .stButton {
        margin-left: 10px;
    }
    .chat-content {
        margin-bottom: 100px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("MoA: 增强大语言模型")
    
    # 初始化 session_state
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'message_keys' not in st.session_state:
        st.session_state['message_keys'] = []

    chat_column, status_column = st.columns([3, 1])
    with chat_column:
        chat_content = st.container()
        with chat_content:
            st.markdown('<div class="chat-content">', unsafe_allow_html=True)
            for i, (msg, key) in enumerate(zip(st.session_state.messages, st.session_state.message_keys)):
                if msg['role'] == 'user':
                    message(msg['content'], is_user=True, key=key)
                else:
                    message(msg['content'], is_user=False, key=key, avatar_style="bottts")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with status_column:
        st.subheader("生成状态")
        status_placeholder = st.empty()
    
    with st.container():
        st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
        with st.form(key='my_form', clear_on_submit=True):
            cols = st.columns([4, 1])
            with cols[0]:
                user_input = st.text_input("What is your question?", key="user_input")
            with cols[1]:
                submit_button = st.form_submit_button("Send")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if submit_button and user_input:
        user_msg = {"role": "user", "content": user_input}
        st.session_state.messages.append(user_msg)
        user_key = f"user_{len(st.session_state.messages)}_{int(time.time() * 1000)}"
        st.session_state.message_keys.append(user_key)
        
        with chat_content:
            message(user_input, is_user=True, key=user_key)
        
        reference_models = [MODEL_REFERENCE_1, MODEL_REFERENCE_2, MODEL_REFERENCE_3]
        references = []
        for i, model in enumerate(reference_models):
            status_placeholder.text(f"正在处理: {model}")
            response = generate_with_references(
                model=model,
                messages=[{"role": "user", "content": user_input}],
                references=[],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                api_base=API_BASE,
                api_key=API_KEY
            )
            reference_output = ""
            for line in response.iter_lines():
                if line:
                    line_decoded = line.decode('utf-8').strip()
                    if line_decoded.startswith('data: {'):
                        json_data = line_decoded[6:]
                        try:
                            chunk = json.loads(json_data)
                            reference_output += process_chunk(chunk)
                        except json.JSONDecodeError:
                            pass
            references.append(reference_output)
        
        status_placeholder.text(f"正在处理: {MODEL_AGGREGATE}")
        final_response = generate_with_references(
            model=MODEL_AGGREGATE,
            messages=[{"role": "user", "content": user_input}],
            references=references,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_base=API_BASE_2,
            api_key=API_KEY_2
        )
        
        full_response = ""
        for line in final_response.iter_lines():
            if line:
                line_decoded = line.decode('utf-8').strip()
                if line_decoded.startswith('data: {'):
                    json_data = line_decoded[6:]
                    try:
                        chunk = json.loads(json_data)
                        content = process_chunk(chunk)
                        full_response += content
                    except json.JSONDecodeError:
                        pass
        
        ai_msg = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(ai_msg)
        ai_key = f"assistant_{len(st.session_state.messages)}_{int(time.time() * 1000)}"
        st.session_state.message_keys.append(ai_key)
        
        with chat_content:
            message(full_response, is_user=False, key=ai_key, avatar_style="bottts")
        
        status_placeholder.text("生成完成")

if __name__ == "__main__":
    main()

