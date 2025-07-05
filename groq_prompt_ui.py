from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

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

# --- Prompt Template ---
prompt_template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:

Explanation Style: {style_input}  
Explanation Length: {length_input}  

1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  

2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length.
"""
)

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
