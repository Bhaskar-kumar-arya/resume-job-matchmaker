# -----------------------------------------------------------------------------
# Streamlit Resume-Job Matcher (Updated with Cross-Encoder Re-ranking)
#
# Description:
# This application uses a two-stage process for matching resumes to jobs.
# 1. Bi-Encoder (SentenceTransformer): Quickly retrieves the top N candidate
#    resumes for each job from a large pool.
# 2. Cross-Encoder (MxbaiRerank): Re-ranks the top N candidates for higher
#    accuracy.
#
# New Features:
# - Optional Cross-Encoder re-ranking for improved accuracy.
# - Optional "Caste Boost" to adjust rankings based on specified criteria.
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from mxbai_rerank import MxbaiRerankV2 # <-- IMPORT THE RERANKER LIBRARY
import torch
import os
import random

# --- CONSTANTS ---
# Using the existing model path for the bi-encoder
# BI_ENCODER_MODEL_PATH = 'Leo1212/longformer-base-4096-sentence-transformers-all-nli-stsb-quora-nq'
BI_ENCODER_MODEL_PATH = './triplet/resume-matcher-longformer'

# Define the cross-encoder model name
CROSS_ENCODER_MODEL_NAME = "mixedbread-ai/mxbai-rerank-large-v2"


# --- ONE-TIME SETUP AND CACHING ---

@st.cache_resource
def load_bi_encoder_model():
    """
    Loads the SentenceTransformer (bi-encoder) model.
    This function will now raise an exception on failure.
    """
    model = SentenceTransformer(BI_ENCODER_MODEL_PATH)
    model.max_seq_length = 1024
    return model

# --- NEW: CACHED FUNCTION TO LOAD THE RERANKER MODEL ---
@st.cache_resource
def load_reranker_model():
    """
    Loads the MxbaiRerank (cross-encoder) model on demand.
    This function will now raise an exception on failure.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MxbaiRerankV2(CROSS_ENCODER_MODEL_NAME, device=device)
    model.max_seq_length = 1024
    return model


# --- CORE LOGIC ---

def find_top_matches(df, top_n, bi_encoder, enable_boost, caste_ratio, boost_percentage, enable_reranker):
    """
    The main matching function with an added re-ranking step.

    Args:
        df (pd.DataFrame): The input data.
        top_n (int): The number of top resumes to find.
        bi_encoder (SentenceTransformer): The loaded bi-encoder model.
        enable_boost (bool): Flag to enable the caste boost.
        caste_ratio (int): The percentage of resumes for 'lower caste'.
        boost_percentage (float): The score boost percentage.
        enable_reranker (bool): Flag to enable the cross-encoder re-ranking step.

    Returns:
        dict: A dictionary of jobs and their top matching resumes.
    """
    df.dropna(subset=['resume_text', 'job_description_text'], inplace=True)

    unique_resumes = df['resume_text'].drop_duplicates().tolist()
    unique_jobs = df['job_description_text'].drop_duplicates().tolist()

    if not unique_resumes or not unique_jobs:
        return {}

    # --- CASTE ASSIGNMENT (No changes here) ---
    caste_map = {}
    if enable_boost:
        for resume_text in unique_resumes:
            if random.randint(1, 100) <= caste_ratio:
                caste_map[resume_text] = 'lower'
            else:
                caste_map[resume_text] = 'higher'

    resume_to_rows_map = {text: (group.index + 2).tolist() for text, group in df.groupby('resume_text')}

    # 1. STAGE 1: Fast retrieval with Bi-Encoder
    with st.spinner("Stage 1: Generating embeddings and finding initial candidates..."):
        resume_embeddings = bi_encoder.encode(unique_resumes, convert_to_tensor=True, show_progress_bar=True)
        job_embeddings = bi_encoder.encode(unique_jobs, convert_to_tensor=True, show_progress_bar=True)
        cosine_scores = util.pytorch_cos_sim(job_embeddings, resume_embeddings)

    # --- NEW: Load reranker model ONLY if needed ---
    reranker = None
    if enable_reranker:
        with st.spinner("Loading the cross-encoder model for re-ranking..."):
            try:
                reranker = load_reranker_model()
                st.toast("Cross-encoder re-ranking model loaded!", icon="✨")
            except Exception as e:
                st.error(f"Could not load the re-ranking model. Proceeding without it. Error: {e}")
                reranker = None # Ensure reranker is None so the next step is skipped


    results = {}
    progress_bar = st.progress(0, text="Processing jobs...")

    for i, job_desc in enumerate(unique_jobs):
        # Initial candidate retrieval
        job_scores = cosine_scores[i]
        top_results = torch.topk(job_scores, k=min(top_n, len(unique_resumes)))

        matches = []
        for score, resume_idx in zip(top_results.values, top_results.indices):
            unique_resume_text = unique_resumes[resume_idx.item()]
            original_row_numbers = resume_to_rows_map[unique_resume_text]

            match_data = {
                "score": score.item(),
                "resume_text": unique_resume_text,
                "rows": original_row_numbers,
                "caste": caste_map.get(unique_resume_text, 'N/A'),
                "boosted": False,
                "reranked": False # Add a flag for re-ranking
            }
            matches.append(match_data)

        # --- CASTE BOOST LOGIC (No changes here) ---
        if enable_boost:
            boost_factor = 1 + (boost_percentage / 100.0)
            for match in matches:
                if match["caste"] == 'lower':
                    match["score"] = min(1.0, match["score"] * boost_factor)
                    match["boosted"] = True
            matches.sort(key=lambda x: x['score'], reverse=True)

        # --- NEW: STAGE 2: Re-ranking with Cross-Encoder ---
        if enable_reranker and reranker is not None:
            # Prepare documents for the reranker
            candidate_resumes = [match['resume_text'] for match in matches]

            # The reranker returns a list of result objects with new scores
            reranked_results = reranker.rank(
                query=job_desc,
                documents=candidate_resumes,
                return_documents=False, # We only need the new scores and order
                top_k=len(candidate_resumes) # Rerank all candidates
            )

            # Create a map of original index to new score
            new_scores = {result.index: result.score for result in reranked_results}

            # Update the original matches with new scores and flag
            for idx, match in enumerate(matches):
                if idx in new_scores:
                    match['score'] = new_scores[idx]
                    match['reranked'] = True

            # Re-sort the list based on the new, more accurate scores
            matches.sort(key=lambda x: x['score'], reverse=True)
        # --- END OF RE-RANKING LOGIC ---

        results[job_desc] = matches
        progress_bar.progress((i + 1) / len(unique_jobs), text=f"Processing jobs... {i+1}/{len(unique_jobs)}")

    progress_bar.empty()
    return results


# --- STREAMLIT UI ---

def main():
    """Defines the Streamlit user interface and application flow."""
    st.set_page_config(page_title="Resume-Job Matcher", layout="wide")

    # Load the primary (bi-encoder) model with UI feedback
    try:
        bi_encoder = load_bi_encoder_model()
        st.toast("Bi-encoder model loaded successfully!", icon="✅")
    except Exception as e:
        st.error(f"Fatal: Could not load the main bi-encoder model. The app cannot continue. Error: {e}")
        st.stop()


    # --- Sidebar for Controls ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="The CSV must contain 'resume_text' and 'job_description_text' columns."
        )
        top_n = st.number_input(
            "Top N Candidates to retrieve",
            min_value=1, max_value=50, value=10, step=1,
            help="The number of initial candidates to find before re-ranking."
        )
        st.markdown("---")

        # --- NEW: RE-RANKING UI ---
        st.header("✨ Re-ranking Settings")
        enable_reranker = st.toggle(
            "Enable Cross-Encoder Re-ranking",
            value=True,
            help="Use a more powerful model to re-rank the top candidates for better accuracy."
        )
        st.markdown("---")

        # --- CASTE BOOST UI (No changes here) ---
        st.header("📈 Score Boost Settings")
        enable_caste_boost = st.toggle(
            "Enable Caste Score Boost",
            value=False,
            help="Enable to apply a score boost to a random subset of resumes(the lower caste people)."
        )
        caste_ratio = st.slider(
            "lower caste ratio (%)", 0, 30, 25,
            help="The percentage of resumes that will be randomly assigned a boost.",
            disabled=not enable_caste_boost
        )
        boost_percentage = st.slider(
            "Score Boost (%)", 0.0, 1.5, 1.0, step=0.01,
            help="The percentage boost to apply to the scores.",
            disabled=not enable_caste_boost
        )

    # --- Main Content Area ---
    st.title("📄↔️💼 Two-Stage Resume-Job Matcher")
    st.markdown("""
        Upload a CSV to find the best resume matches for each job description.
        - **Stage 1 (Fast Retrieval):** A `SentenceTransformer` model quickly finds the top candidate resumes.
        - **Stage 2 (Accurate Re-ranking):** A powerful `Cross-Encoder` model then re-ranks these candidates for higher accuracy.
    """)
    st.markdown("---") # Added a separator for visual clarity

    # --- MOVED: Button is now in the main area ---
    process_button = st.button("Find Matches", type="primary", use_container_width=True)

    if process_button and uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'resume_text' not in df.columns or 'job_description_text' not in df.columns:
                st.error("Error: CSV must contain 'resume_text' and 'job_description_text' columns.")
            else:
                match_results = find_top_matches(df, top_n, bi_encoder, enable_caste_boost, caste_ratio, boost_percentage, enable_reranker)
                st.success(f"✅ Matching complete! Found results for {len(match_results)} unique jobs.")

                if not match_results:
                    st.warning("No valid resume/job pairs found in the uploaded file.")
                else:
                    for job_desc, matches in match_results.items():
                        with st.expander(f"▶ **JOB:** {job_desc[:100]}..."):
                            st.subheader("Full Job Description")
                            st.markdown(f"> {job_desc}")
                            st.markdown("---")
                            st.subheader(f"Top {len(matches)} Matching Resumes")

                            for i, match in enumerate(matches):
                                # --- MODIFIED: DISPLAY BOOST/RE-RANK INFO ---
                                rank_text = f"**Rank {i+1}**"
                                score_text = ""
                                if match.get('reranked', False):
                                    # Cross-encoder scores are not percentages, they are logits.
                                    score_text = f"**Re-rank Score: {match['score']:.4f}** ✨"
                                else:
                                    score_text = f"**Match Score: {match['score']*100:.2f}%**"

                                if match.get('boosted', False):
                                    score_text += " 📈 (Boost Applied)"

                                col1, col2 = st.columns([1, 4])
                                col1.markdown(rank_text)
                                col2.markdown(score_text)

                                st.info(f"**Resume:** {match['resume_text']}")
                                st.caption(f"Original CSV Row(s): {match['rows']}")
                                st.markdown("---")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

    elif process_button and uploaded_file is None:
        st.warning("Please upload a CSV file first.")

if __name__ == "__main__":
    main()