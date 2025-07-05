from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

# Load environment variables
load_dotenv()

# Initialize the Groq model (use your actual model name if needed like "mixtral-8x7b-32768")
model = ChatGroq(model_name="llama3-70b-8192")

st.set_page_config(page_title="Research Paper Summarizer")
st.header("📄 Research Paper Summarization Tool")

# --- User Inputs ---
paper_input = st.selectbox(
    "Select Research Paper Name", 
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers", 
        "GPT-3: Language Models are Few-Shot Learners", 
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style", 
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length", 
    ["Short (1–2 paragraphs)", "Medium (3–5 paragraphs)", "Long (detailed explanation)"]
)

prompt_template = load_prompt('prompt_temp.json')

# --- Summary Button ---
if st.button("Summarize"):
    filled_prompt = prompt_template.format(
        paper_input=paper_input,
        style_input=style_input,
        length_input=length_input
    )

    with st.spinner("Generating summary..."):
        response = model.invoke(filled_prompt)
        st.subheader("📋 Summary")
        st.write(response.content)
