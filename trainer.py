# ----------------------------------------------------------------------
# Fine-tune SBERT on Resume-Job Dataset for Cosine Similarity Search
# ----------------------------------------------------------------------

# Step 1: Install required packages
!pip install -U sentence-transformers datasets

# Step 2: Import libraries
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random
from google.colab import drive

# Step 3: Mount Google Drive
drive.mount('/content/drive')

# Step 4: Load the dataset
dataset = load_dataset("cnamuangtoun/resume-job-description-fit")
train_data = dataset["train"]

# Step 5: Convert string labels to normalized float scores for similarity
# We map the labels to a 0.0-1.0 scale where 1.0 is a perfect match.
label_map = {"No Fit": 0.0, "Potential Fit": 0.5, "Good Fit": 1.0}

train_examples = []
for row in train_data:
    resume = row['resume_text']
    job_desc = row['job_description_text']
    label = label_map[row['label']]
    train_examples.append(InputExample(texts=[resume, job_desc], label=label))

# Step 6: Shuffle the training examples
random.shuffle(train_examples)

# Step 7: Load a pre-trained SBERT model
# 'all-MiniLM-L6-v2' is a good starting point - it's fast and performs well.
# You can also use 'sentence-transformers/stsb-bert-large' as you had before.
model = SentenceTransformer('sentence-transformers/stsb-bert-large')

# Step 8: Create the DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Step 9: Define the loss function for similarity learning
# CosineSimilarityLoss is ideal for making the cosine similarity between
# the embeddings of the resume and job description match the label score.
train_loss = losses.CosineSimilarityLoss(model=model)

# Step 10: Configure training parameters
warmup_steps = int(len(train_dataloader) * 0.1)  # 10% of train data for warm-up
epochs = 4

# Step 11: Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    output_path="./resume_job_sbert_temp",  # temporary local save
    use_amp=True
)

# Step 12: Save the fine-tuned model to your Google Drive
drive_model_path = "/content/drive/MyDrive/resume_job_sbert_model"
model.save(drive_model_path)

print("\n---------------------------------")
print("✅ Training complete!")
print(f"Model saved to Google Drive at: {drive_model_path}")
print("---------------------------------")

# ----------------------------------------------------------------------
# You can later load the model from your Drive with:
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("/content/drive/MyDrive/resume_job_sbert_model")
# ----------------------------------------------------------------------
