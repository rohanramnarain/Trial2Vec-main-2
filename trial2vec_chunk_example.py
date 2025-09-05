import pandas as pd
import torch
from trial2vec import Trial2Vec
import os

##########################
# 1) Initialize the Model
##########################

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Ensures CPU-only inference
model = Trial2Vec(device="cpu")
model.from_pretrained('trial_search/pretrained_trial2vec')

##############################
# 2) Define a Chunking Function
##############################
def chunk_text(text, chunk_size=150, overlap=20):
    """
    Splits the given text into smaller chunks of length `chunk_size` words,
    with an optional `overlap` between consecutive chunks.
    """
    if not isinstance(text, str):
        text = str(text)
    
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        if end > len(words):
            end = len(words)
        chunk = words[start:end]
        chunks.append(" ".join(chunk))

        # Move start forward so we don't get stuck
        start += (chunk_size - overlap)
        if start <= 0:
            start += 1

    return chunks

##########################
# 3) Load and Combine the 9 CSVs
##########################
csv_files = [
    "NCT03059992.csv",
"NCT03257033.csv",
"NCT03363841.csv",
"NCT03417102.csv",
"NCT03425279.csv",
"NCT03464019.csv",
"NCT03504488.csv",
"NCT04003467.csv",
"NCT04153149.csv",
"NCT04163185.csv",
"NCT04180488.csv",
"NCT04524273.csv",
"NCT04544293.csv",
"NCT05059223.csv",
"NCT05085366.csv",
"NCT05109091.csv",
"NCT05208944.csv",
"NCT05291091.csv",
"NCT05357950.csv",
"NCT05481879.csv",
"NCT05693142.csv",
"NCT05696717.csv",
"NCT05731544.csv",
"NCT06079736.csv",
"NCT06185673.csv",
"NCT06204809.csv",
"NCT06255782.csv",
"NCT06377930.csv",
"NCT06601192.csv",
"NCT06663137.csv",
"NCT06736509.csv",
"NCT05579249.csv",
"NCT06034275.csv",
"NCT05065411.csv",
"NCT05424276.csv",
"NCT04887298.csv",
"NCT04073498.csv",
"NCT03959527.csv",
"NCT04469595.csv",
"NCT04370054.csv",
]

df_list = []
for csv_file in csv_files:
    # Load the CSV
    temp_df = pd.read_csv(csv_file)
    # Rename columns so we can properly match them in the code below
    temp_df = temp_df.rename(
        columns={
            "NCT Number": "nct_id",
            "Study Title": "title",
            "Brief Summary": "brief_summary",
            "Conditions": "condition",
            "Interventions": "intervention",
            "Primary Outcome Measures": "primary_outcome",
            "Secondary Outcome Measures": "secondary_outcome",
            "Sponsor": "sponsor",
            "Collaborators": "collaborators",
            "Sex": "sex",
            "Age": "age",
            "Phases": "phase",
            "Enrollment": "enrollment",
            "Funder Type": "funder_type",
            "Study Type": "study_type",
            "Study Design": "study_design",
            "Other IDs": "other_ids",
            "Start Date": "start_date",
            "Completion Date": "completion_date",
            "Locations": "locations"
            # If there are other columns you need, add them here
        }
    )
    df_list.append(temp_df)

# Combine into one DataFrame
all_df = pd.concat(df_list, ignore_index=True)

##########################
# 4) Process Each Row (All Trials)
##########################
output_rows = []
chunk_details = []

for idx, row in all_df.iterrows():
    # Safely fetch columns; if a column is missing, fallback to empty string
    nct_id = row.get("nct_id", "")
    title = row.get("title", "")
    brief_summary = row.get("brief_summary", "")
    condition = row.get("condition", "")
    intervention = row.get("intervention", "")
    primary_outcome = row.get("primary_outcome", "")
    secondary_outcome = row.get("secondary_outcome", "")
    sponsor = row.get("sponsor", "")
    collaborators = row.get("collaborators", "")
    sex = row.get("sex", "")
    age = row.get("age", "")
    phase = row.get("phase", "")
    enrollment = row.get("enrollment", "")
    funder_type = row.get("funder_type", "")
    study_type = row.get("study_type", "")
    study_design = row.get("study_design", "")
    other_ids = row.get("other_ids", "")
    start_date = row.get("start_date", "")
    completion_date = row.get("completion_date", "")
    locations = row.get("locations", "")

    # Combine these fields into one text block
    combined_text = f"""
    Title: {title}
    Brief Summary: {brief_summary}
    Condition: {condition}
    Intervention: {intervention}
    Primary Outcome: {primary_outcome}
    Secondary Outcome: {secondary_outcome}
    Sponsor: {sponsor}
    Collaborators: {collaborators}
    Sex: {sex}
    Age: {age}
    Phase: {phase}
    Enrollment: {enrollment}
    Funder Type: {funder_type}
    Study Type: {study_type}
    Study Design: {study_design}
    Other IDs: {other_ids}
    Start Date: {start_date}
    Completion Date: {completion_date}
    Locations: {locations}
    """

    # 4b) Chunk the text
    chunks = chunk_text(combined_text, chunk_size=150, overlap=20)
    print(f"Created {len(chunks)} chunks for trial {nct_id}")

    # 4c) Embed each chunk
    chunk_embeddings_list = []
    for i, chunk_text_ in enumerate(chunks):
        with torch.inference_mode():
            # Pass a list with a single string
            single_emb = model.sentence_vector([chunk_text_])
            # single_emb has shape [1, embed_dim], grab index 0 for the vector
            single_emb = single_emb[0]

        # Save chunk details (optional chunk-level embeddings)
        chunk_dict = {
            "nct_id": nct_id,
            "chunk_id": f"{nct_id}_chunk_{i}",
            "chunk_text": chunk_text_
        }
        emb_as_list = single_emb.tolist()
        for dim_idx, val in enumerate(emb_as_list):
            chunk_dict[f"emb_{dim_idx}"] = val
        chunk_details.append(chunk_dict)

        # Accumulate for final average
        chunk_embeddings_list.append(single_emb)

    # 4d) Average chunk embeddings
    if len(chunk_embeddings_list) == 0:
        # If no text at all, just use a zero-vector
        trial_embedding_tensor = torch.zeros(model.vector_size)
    elif len(chunk_embeddings_list) == 1:
        trial_embedding_tensor = chunk_embeddings_list[0]
    else:
        chunk_embeddings_tensor = torch.stack(chunk_embeddings_list, dim=0)
        trial_embedding_tensor = chunk_embeddings_tensor.mean(dim=0)

    # 4e) Convert final embedding to a dictionary
    emb_dict = {"nct_id": nct_id}
    trial_emb_list = trial_embedding_tensor.tolist()
    for dim_idx, val in enumerate(trial_emb_list):
        emb_dict[f"emb_{dim_idx}"] = val
    output_rows.append(emb_dict)

##########################
# 5) Save Final CSVs
##########################

# Per-trial final embeddings (128D)
emb_df = pd.DataFrame(output_rows)
emb_df.to_csv("all_trials_embeddings.csv", index=False)

# Optional: chunk-level embeddings
chunk_df = pd.DataFrame(chunk_details)
chunk_df.to_csv("all_trials_chunk_embeddings.csv", index=False)

print("Done! All embeddings saved to 'all_trials_embeddings.csv' and 'all_trials_chunk_embeddings.csv'.")
