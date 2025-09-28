
import os

# Set the run name before calling model.fit
os.environ["WANDB_DISABLED"] = "true"
import pandas as pd
import random
from tqdm.autonotebook import tqdm
import torch
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------
# Step 1: Function to Create Training Triplets
# ----------------------------------------------------------------------
def create_training_triplets(dataset, negatives_per_positive: int = 1):
    """
    Reads the dataset and creates training triplets for TripletLoss.
    The triplet format is (anchor=JD, positive=GoodFit_Resume, negative=NoFit_Resume).

    Args:
        dataset: The Hugging Face dataset object.
        negatives_per_positive: The number of negative resumes to sample for each positive resume.

    Returns:
        A list of InputExample objects for triplet training.
    """
    print("🧠 Preparing data and creating triplets...")
    df = dataset.to_pandas()
    df_filtered = df[df['label'].isin(['Good Fit', 'No Fit'])].drop_duplicates()

    resumes_by_jd = {}
    for _, row in df_filtered.iterrows():
        jd = row['job_description_text']
        resume = row['resume_text']
        label = row['label']

        if not jd or not resume:
            continue

        if jd not in resumes_by_jd:
            resumes_by_jd[jd] = {'Good Fit': set(), 'No Fit': set()}
        
        resumes_by_jd[jd][label].add(resume)

    train_examples = []
    for jd, resumes in tqdm(resumes_by_jd.items(), desc="Generating Triplets"):
        positive_resumes = list(resumes['Good Fit'])
        negative_resumes = list(resumes['No Fit'])

        if not positive_resumes or not negative_resumes:
            continue

        for pos_resume in positive_resumes:
            num_to_sample = min(negatives_per_positive, len(negative_resumes))
            if num_to_sample == 0:
                continue
                
            sampled_negatives = random.sample(negative_resumes, num_to_sample)
            
            for neg_resume in sampled_negatives:
                train_examples.append(InputExample(texts=[jd, pos_resume, neg_resume]))

    print(f"✅ Generated {len(train_examples)} unique triplets.")
    return train_examples

# ----------------------------------------------------------------------
# Main Training Script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --- Configuration ---
    model_name = 'Leo1212/longformer-base-4096-sentence-transformers-all-nli-stsb-quora-nq'
    train_batch_size = 2      # Adjust based on your GPU VRAM
    num_epochs = 3            # Number of epochs to train for
    negatives_per_positive = 4 # Number of negative samples per positive
    
    # ✔️ Correct, writable paths for Kaggle environment
    # The final model will be saved here after training completes.
    output_path = "/kaggle/working/resume-matcher-longformer"
    # Checkpoints will be saved here during training.
    checkpoint_path = "/kaggle/working/checkpoints/"

    # --- Step 1: Load and Prepare Data ---
    dataset = load_dataset("cnamuangtoun/resume-job-description-fit", split="train")
    train_examples = create_training_triplets(dataset, negatives_per_positive=negatives_per_positive)

    # --- Step 2: Define Model ---
    # A single model learns a shared embedding space for both resumes and JDs.
    print("\n✨ Initializing a single Sentence Transformer model...")
    model = SentenceTransformer(model_name)
    model.max_seq_length = 2048

    # --- Step 3: Prepare DataLoader and Loss Function ---
    print("📦 Setting up DataLoader and TripletLoss...")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.TripletLoss(model=model)

    # --- Step 4: Train the Model with Auto-Resuming ---
    # The `fit` method automatically detects and resumes from the latest checkpoint
    # in the `checkpoint_path` directory.
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) # 10% of train data for warm-up
    
    print(f"🚀 Starting training for {num_epochs} epoch(s)...")
    print(f"💾 Final model will be saved to: {output_path}")
    print(f"🔄 Checkpoints will be saved to and loaded from: {checkpoint_path}")
    
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              output_path=output_path,
              show_progress_bar=True,
              checkpoint_path=checkpoint_path,
              checkpoint_save_steps=500, # Save a checkpoint every 500 training steps
              )

    print(f"\n🎉 Training complete!")