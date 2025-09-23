# app.py
# -----------------------------------------------------------------------------
# This script creates a simple web application using Streamlit to demonstrate
# semantic matching between a resume and a job description. It uses a
# pre-trained Sentence-BERT model from the Hugging Face library to generate
# embeddings and calculate a contextual similarity score.
#
# To Run:
# 1. Make sure you have the required libraries installed:
#    pip install streamlit sentence-transformers torch
# 2. Save this code as 'app.py'.
# 3. Open a terminal in the same directory and run the command:
#    streamlit run app.py
# -----------------------------------------------------------------------------

import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Internship Matchmaker",
    page_icon="🤖",
    layout="wide"
)

# --- MODEL LOADING ---
# Use a caching decorator to load the model only once, improving performance.
@st.cache_resource
def load_model():
    """Loads the Sentence-BERT model from Hugging Face."""
    # model = SentenceTransformer('./resume_job_sbert_model')
    model = SentenceTransformer('Leo1212/longformer-base-4096-sentence-transformers-all-nli-stsb-quora-nq')
    return model

# --- MAIN APP ---
def main():
    # --- HEADER ---
    st.title("AI Internship Matchmaker POC 🤖")
    st.markdown("This Proof of Concept demonstrates the power of **semantic search** for matching candidates to internships. Instead of just matching keywords, it understands the *contextual meaning* of the text.")

    # Load the AI model
    with st.spinner("Loading AI model... This might take a moment on first run."):
        model = load_model()

    # --- INPUT COLUMNS ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Candidate's Resume")
        # Pre-filled example text to guide the user
        resume_text = st.text_area(
            "Paste the full text of the resume here:",
            height=300,
            value="""
John Doe - Data Science Enthusiast

Experience:
Project on Customer Behavior Analysis (College Project)
- Developed a predictive model using Python, Pandas, and Scikit-learn to analyze customer purchasing patterns.
- Engineered features to identify key drivers of user retention.
- My goal was to forecast which clients were likely to stop using our service.

Skills: Python, Machine Learning, Data Analysis, SQL
            """,
            help="Provide as much detail as possible from the resume's experience and skills sections."
        )

    with col2:
        st.subheader("Internship Description")
        # Pre-filled example text to show a non-obvious match
        job_description_text = st.text_area(
            "Paste the full text of the internship description here:",
            height=300,
            value="""
Data Analyst Intern

Responsibilities:
- Join our marketing analytics team to work on a vital customer attrition project.
- You will be responsible for exploring datasets to understand why subscribers leave.
- The role requires building a system to flag at-risk customers proactively.

Requirements: Experience with data manipulation libraries, basic understanding of predictive modeling.
            """,
            help="Include the responsibilities and requirements of the role."
        )

    # --- ACTION & OUTPUT ---
    st.divider()
    
    # Center the button using columns
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        if st.button("✨ Calculate Match Score", type="primary", use_container_width=True):
            if resume_text and job_description_text:
                with st.spinner("Analyzing texts and calculating score..."):
                    # 1. Encode the texts into numerical vectors (embeddings)
                    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
                    job_embedding = model.encode(job_description_text, convert_to_tensor=True)

                    # 2. Calculate the cosine similarity between the vectors
                    # The result is a tensor, so we extract the score with .item()
                    cosine_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
                    
                    # 3. Format the score as a percentage
                    match_percentage = round(cosine_score * 100, 2)

                    # --- Display Results ---
                    st.header("Results")
                    st.metric(
                        label="Semantic Similarity Score",
                        value=f"{match_percentage}%",
                        delta=None # No delta needed
                    )
                    
                    st.progress(match_percentage / 100)

                    # Provide a simple explanation based on the score
                    if match_percentage >= 75:
                        st.success("✅ **Strong Match:** The candidate's experience and the internship's requirements are highly aligned in meaning.")
                    elif match_percentage >= 50:
                        st.info("🟡 **Moderate Match:** There is a good degree of contextual overlap. Worth reviewing.")
                    else:
                        st.warning("❌ **Low Match:** The resume and job description seem to cover different concepts and skills.")

                    # Explanation of the 'magic'
                    st.markdown("""
                    **How does this work?** The AI model converted both texts into 'meaning vectors'. The score reflects how closely aligned these vectors are. This is why it correctly identifies that a *"customer churn project"* in the resume is a great fit for a *"customer attrition project"* in the job description, even though the keywords are different.
                    """)
            else:
                st.error("Please ensure both text boxes are filled before calculating the score.")

if __name__ == "__main__":
    main()