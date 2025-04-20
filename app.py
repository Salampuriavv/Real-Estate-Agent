import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# ---------- CONFIG ----------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image):
    raw_image = Image.open(image).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

llm = ChatOpenAI(temperature=0)

agent1_prompt = PromptTemplate.from_template(
    "You are a property inspection assistant. Given an image caption and user query, detect issues and suggest fixes.\n\nImage Caption: {caption}\nUser Text: {text}\nAnswer:"
)
agent1_chain = agent1_prompt | llm | StrOutputParser()

agent2_prompt = PromptTemplate.from_template(
    "You are a legal assistant for tenancy-related questions. Provide jurisdiction-specific, helpful answers.\n\nUser Question: {input}\nAnswer:"
)
agent2_chain = agent2_prompt | llm | StrOutputParser()

def route_agent(image, text):
    if image is not None:
        return "agent1"
    keywords = ["notice", "evict", "deposit", "landlord", "tenant", "rent", "contract"]
    if any(word in text.lower() for word in keywords):
        return "agent2"
    return "ask"

# ---------- UI STYLING ----------
st.set_page_config(page_title="🏠 AI Property Assistant", layout="centered")

st.markdown("""
    <style>
        .main {
            background-color: #f7f9fb;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .title {
            font-size: 2.5rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 1rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .chat-bubble {
            border-radius: 16px;
            padding: 1rem;
            margin: 0.5rem 0;
            max-width: 90%;
            word-wrap: break-word;
        }
        .user-msg {
            background-color: #d4eaf7;
            align-self: flex-end;
            margin-left: auto;
        }
        .bot-msg {
            background-color: #e8f5e9;
            align-self: flex-start;
            margin-right: auto;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🏡 Real Estate AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Report property issues or ask legal rental questions</div>', unsafe_allow_html=True)

# ---------- THEME SWITCH ----------
dark_mode = st.toggle("🌙 Dark Mode")
if dark_mode:
    st.markdown(
        """
        <style>
            body {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            .chat-bubble {
                color: #000;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------- INPUT ----------
with st.container():
    with st.expander("💬 Ask your question or upload an image", expanded=True):
        image = st.file_uploader("📸 Upload a property image (optional)", type=["png", "jpg", "jpeg"])
        text_input = st.text_area("✍️ Describe your concern or ask your question:", height=150)

        submit = st.button("🚀 Submit")

# ---------- CHAT DISPLAY ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if submit:
    if not text_input and not image:
        st.warning("⚠️ Please enter a message or upload an image.")
    else:
        agent = route_agent(image, text_input)

        # Show user message
        st.session_state.chat_history.append({
            "role": "user",
            "avatar": "🧑",
            "text": text_input
        })

        if agent == "agent1":
            st.info("🛠️ Routing to Property Assistant...")
            with st.spinner("Analyzing image and message..."):
                caption = caption_image(image)
                result = agent1_chain.invoke({"caption": caption, "text": text_input})
                st.session_state.chat_history.append({
                    "role": "bot",
                    "avatar": "🛠️",
                    "text": result
                })

        elif agent == "agent2":
            st.info("📄 Routing to Legal Assistant...")
            with st.spinner("Checking tenancy law guidance..."):
                result = agent2_chain.invoke({"input": text_input})
                st.session_state.chat_history.append({
                    "role": "bot",
                    "avatar": "📄",
                    "text": result
                })

        else:
            st.warning("🤖 Not sure how to help. Try rephrasing your question.")
            st.session_state.chat_history.append({
                "role": "bot",
                "avatar": "🤖",
                "text": "Sorry, I couldn't identify the nature of your question. Is it about a property issue or rental law?"
            })

# ---------- DISPLAY CHAT ----------
st.markdown("---")
st.subheader("🗨️ Chat History")
for msg in st.session_state.chat_history:
    css_class = "chat-bubble user-msg" if msg["role"] == "user" else "chat-bubble bot-msg"
    st.markdown(f"""
        <div class="{css_class}">
            <strong>{msg["avatar"]}</strong> {msg["text"]}
        </div>
    """, unsafe_allow_html=True)
