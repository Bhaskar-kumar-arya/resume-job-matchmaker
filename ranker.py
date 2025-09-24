# -----------------------------------------------------------------------------
# Streamlit Resume-Job Matcher (Updated)
#
# Description:
# This application provides a web interface to perform a many-to-many matching
# between jobs and resumes from an uploaded CSV file. It identifies unique job
# descriptions and resumes, calculates their semantic similarity using a
# SentenceTransformer model, and displays the top N matching resumes for each job.
#
# New Feature:
# Includes an optional "Caste Boost" system. When enabled, a specified ratio
# of resumes are randomly assigned a 'lower caste' status and receive a
# configurable score boost to promote them in the rankings.
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import random

if torch.cuda.is_available():
    print(f"✅ GPU is available and ready to use!")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("❌ GPU not found. PyTorch will use the CPU.")

    # --- CONSTANTS ---
# The path where the model will be stored/loaded from.
MODEL_PATH = 'Leo1212/longformer-base-4096-sentence-transformers-all-nli-stsb-quora-nq'
# The default model to download if a local model is not found.
DEFAULT_MODEL = 'all-MiniLM-L6-v2'


    # --- ONE-TIME SETUP AND CACHING ---

def setup_model_path():
    """
    Checks if the model exists. If not, it downloads and saves a default model.
    This ensures the app is runnable out-of-the-box.
    """
    pass
    # if not os.path.exists(MODEL_PATH):
    #     st.toast(f"Model not found at '{MODEL_PATH}'. Downloading a default model...")
    #     print(f"Model not found at '{MODEL_PATH}'. Downloading '{DEFAULT_MODEL}'...")
    #     try:
    #         model = SentenceTransformer(DEFAULT_MODEL)
    #         model.save(MODEL_PATH)
    #         print("Default model downloaded and saved successfully.")
    #         st.toast("Default model downloaded!")
    #     except Exception as e:
    #         st.error(f"Failed to download default model. Please check your internet connection. Error: {e}")
    #         st.stop()

@st.cache_resource
def load_model():
    """
    Loads the SentenceTransformer model from the specified path.
    Uses Streamlit's caching to load the model only once.
    """
    try:
        model = SentenceTransformer(MODEL_PATH)
        model.max_seq_length = 1024
        return model
    except Exception as e:
        st.error(f"Error loading the model from '{MODEL_PATH}'. Ensure the model exists and is compatible. Error: {e}")
        st.stop()


# --- CORE LOGIC ---

def find_top_matches(df, top_n, model, enable_boost, caste_ratio, boost_percentage):
    """
    The main matching function. It takes the DataFrame, number of matches, and
    the loaded model to perform the similarity search.

    Args:
        df (pd.DataFrame): The input data with 'resume_text' and 'job_description_text'.
        top_n (int): The number of top resumes to find for each job.
        model (SentenceTransformer): The loaded sentence-transformer model.
        enable_boost (bool): Flag to enable/disable the caste boost.
        caste_ratio (int): The percentage of resumes to be assigned 'lower caste'.
        boost_percentage (float): The score boost percentage for lower caste resumes.

    Returns:
        dict: A dictionary where keys are job descriptions and values are lists of
              top matching resumes, each with its score and original location.
    """
    df.dropna(subset=['resume_text', 'job_description_text'], inplace=True)

    # 1. Prepare unique data pools to avoid redundant computations
    print("dropping duplicates")
    unique_resumes = df['resume_text'].drop_duplicates().tolist()
    unique_jobs = df['job_description_text'].drop_duplicates().tolist()

    if not unique_resumes or not unique_jobs:
        return {}

    # --- MODIFICATION: CASTE ASSIGNMENT ---
    caste_map = {}
    if enable_boost:
        st.toast(f"Caste boost enabled. Assigning caste to {len(unique_resumes)} unique resumes...")
        for resume_text in unique_resumes:
            # Assign caste based on the ratio provided by the user
            if random.randint(1, 100) <= caste_ratio:
                caste_map[resume_text] = 'lower'
            else:
                caste_map[resume_text] = 'higher'
    # --- END MODIFICATION ---

    # Map each unique resume back to its original row numbers for reporting
    # The +2 accounts for the header row and 0-based indexing
    resume_to_rows_map = {text: (group.index + 2).tolist() for text, group in df.groupby('resume_text')}

    # 2. Generate embeddings in batches
    with st.spinner("Generating embeddings for all unique resumes and jobs... This may take a moment."):
        print("\nGenerating embeddings for all unique resumes...")
        resume_embeddings = model.encode(unique_resumes, convert_to_tensor=True, show_progress_bar=True)
        
        print("\nGenerating embeddings for all unique job descriptions...")
        job_embeddings = model.encode(unique_jobs, convert_to_tensor=True, show_progress_bar=True)

    # 3. Calculate cosine similarity between all jobs and all resumes
    cosine_scores = util.pytorch_cos_sim(job_embeddings, resume_embeddings)

    # 4. Process results for each job
    results = {}
    for i, job_desc in enumerate(unique_jobs):
        job_scores = cosine_scores[i]
        
        # Get the top N scores and their corresponding indices in the unique_resumes list
        top_results = torch.topk(job_scores, k=min(top_n, len(unique_resumes)))

        matches = []
        for score, resume_idx in zip(top_results.values, top_results.indices):
            unique_resume_text = unique_resumes[resume_idx.item()]
            original_row_numbers = resume_to_rows_map[unique_resume_text]
            
            match_data = {
                "score": score.item(),
                "resume_text": unique_resume_text,
                "rows": original_row_numbers,
                "caste": caste_map.get(unique_resume_text, 'N/A'), # Get assigned caste
                "boosted": False # Flag to track if boost was applied
            }
            matches.append(match_data)
        
        # --- MODIFICATION: APPLY BOOST AND RE-SORT ---
        if enable_boost:
            boost_factor = 1 + (boost_percentage / 100.0)
            for match in matches:
                if match["caste"] == 'lower':
                    original_score = match["score"]
                    boosted_score = original_score * boost_factor
                    # Cap the score at 1.0 to prevent scores > 100%
                    match["score"] = min(1.0, boosted_score)
                    match["boosted"] = True
            
            # Re-sort the list based on the new scores
            matches.sort(key=lambda x: x['score'], reverse=True)
        # --- END MODIFICATION ---

        results[job_desc] = matches
        
    return results


# --- STREAMLIT UI ---

def main():
    """Defines the Streamlit user interface and application flow."""
    st.set_page_config(page_title="Resume-Job Matcher", layout="wide")

    # --- Initial Setup ---
    setup_model_path()
    model = load_model()

    # --- Sidebar for Controls ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="The CSV must contain 'resume_text' and 'job_description_text' columns."
        )
        top_n = st.number_input(
            "Top N Matches to find",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            help="The number of best-matching resumes to show for each job."
        )

        st.markdown("---")
        
        # --- MODIFICATION: CASTE BOOST UI ---
        st.header("✨ Caste Boost Settings")
        enable_caste_boost = st.toggle(
            "Enable Caste Boost",
            value=False,
            help="Enable to apply a score boost to a random subset of resumes."
        )

        caste_ratio = 0
        boost_percentage = 0.0
        
        if enable_caste_boost:
            caste_ratio = st.slider(
                "Lower Caste Ratio (%)",
                min_value=0,
                max_value=30,
                value=25,
                help="The percentage of resumes that will be randomly assigned 'lower caste' status."
            )
            boost_percentage = st.slider(
                "Score Boost (%)",
                min_value=0.0,
                max_value=1,
                value=0.6,
                step=0.1,
                help="The percentage boost to apply to the scores of 'lower caste' resumes."
            )
        # --- END MODIFICATION ---
        
        process_button = st.button("Find Matches", type="primary")

    # --- Main Content Area ---
    st.title("📄↔️💼 Resume-Job Matcher")
    st.markdown("""
        Upload a CSV file containing job descriptions and resumes to find the best matches.
        The tool uses semantic similarity to compare texts.
    """)

    if process_button and uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'resume_text' not in df.columns or 'job_description_text' not in df.columns:
                st.error("Error: CSV must contain 'resume_text' and 'job_description_text' columns.")
            else:
                with st.spinner("Processing your file and finding the best matches..."):
                    print("starting to find top matches")
                    # Pass the new boost parameters to the core function
                    match_results = find_top_matches(df, top_n, model, enable_caste_boost, caste_ratio, boost_percentage)

                st.success(f"✅ Matching complete! Found results for {len(match_results)} unique jobs.")

                # Display results
                if not match_results:
                    st.warning("No valid resume/job pairs found in the uploaded file.")
                else:
                    for job_desc, matches in match_results.items():
                        # The expander label is still truncated for a cleaner look
                        with st.expander(f"▶ **JOB:** {job_desc[:100]}..."):
                            
                            st.subheader("Full Job Description")
                            st.markdown(job_desc)
                            st.markdown("---")
                            st.subheader(f"Top {len(matches)} Matching Resumes")
                            
                            for i, match in enumerate(matches):
                                # --- MODIFICATION: DISPLAY BOOST INFO ---
                                score_display_text = f"**Rank {i+1}: Match Score {match['score']*100:.2f}%**"
                                if match.get('boosted', False):
                                    score_display_text += " 📈 (Boost Applied)"
                                # --- END MODIFICATION ---
                                    
                                st.markdown(score_display_text)
                                st.info(f"**Resume:** {match['resume_text']}")
                                st.caption(f"Original CSV Row(s): {match['rows']}")
                                st.markdown("---")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    elif process_button and uploaded_file is None:
        st.warning("Please upload a CSV file first.")
    else:
        st.info("Upload a file and click 'Find Matches' to begin.")

if __name__ == "__main__":
    main()