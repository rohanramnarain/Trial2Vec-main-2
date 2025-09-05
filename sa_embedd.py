import pandas as pd
import torch
from trial2vec import Trial2Vec
import os

##########################
# 1) Initialize the Model
##########################

os.environ["CUDA_VISIBLE_DEVICES"] = ""     # Force CPU
model = Trial2Vec(device="cpu")
model.from_pretrained("trial_search/pretrained_trial2vec")

##############################
# 2) Chunk‑making helper
##############################
def chunk_text(text, chunk_size=150, overlap=20):
    if not isinstance(text, str):
        text = str(text)

    words, chunks, start = text.split(), [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += max(chunk_size - overlap, 1)
    return chunks

##########################
# 3) Load ONE “mega” CSV
##########################

master_csv = "ctg-studies-completed.csv"         # ⇦ put the file in the same folder or give a full path
all_df = pd.read_csv(master_csv)

# Harmonise column names so the rest of the code still works
rename_map = {
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
    "Locations": "locations",
}
all_df = all_df.rename(columns=rename_map)

##########################
# 4) Embed every trial row
##########################
output_rows, chunk_details = [], []

for idx, row in all_df.iterrows():
    # --- pull out any columns you care about (defaults to "") -------------
    nct_id            = row.get("nct_id", "")
    title             = row.get("title", "")
    brief_summary     = row.get("brief_summary", "")
    condition         = row.get("condition", "")
    intervention      = row.get("intervention", "")
    primary_outcome   = row.get("primary_outcome", "")
    secondary_outcome = row.get("secondary_outcome", "")
    sponsor           = row.get("sponsor", "")
    collaborators     = row.get("collaborators", "")
    sex               = row.get("sex", "")
    age               = row.get("age", "")
    phase             = row.get("phase", "")
    enrollment        = row.get("enrollment", "")
    funder_type       = row.get("funder_type", "")
    study_type        = row.get("study_type", "")
    study_design      = row.get("study_design", "")
    other_ids         = row.get("other_ids", "")
    start_date        = row.get("start_date", "")
    completion_date   = row.get("completion_date", "")
    locations         = row.get("locations", "")
    # ----------------------------------------------------------------------

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

    # 4b) chunk + embed
    chunks = chunk_text(combined_text, chunk_size=150, overlap=20)
    chunk_embeddings_list = []
    for i, chunk_text_ in enumerate(chunks):
        with torch.inference_mode():
            emb = model.sentence_vector([chunk_text_])[0]  # shape [128]
        chunk_dict = {
            "nct_id": nct_id,
            "chunk_id": f"{nct_id}_chunk_{i}",
            "chunk_text": chunk_text_,
            **{f"emb_{j}": v for j, v in enumerate(emb.tolist())},
        }
        chunk_details.append(chunk_dict)
        chunk_embeddings_list.append(emb)

    # 4c) average per‑trial
    if not chunk_embeddings_list:
        trial_emb = torch.zeros(model.vector_size)
    else:
        trial_emb = torch.stack(chunk_embeddings_list).mean(0)

    output_rows.append({
        "nct_id": nct_id,
        **{f"emb_{j}": v for j, v in enumerate(trial_emb.tolist())},
    })

##########################
# 5) Save the outputs
##########################
pd.DataFrame(output_rows).to_csv("completed_sa_all_trials_embeddings.csv", index=False)
pd.DataFrame(chunk_details).to_csv("completed_sa_all_trials_chunk_embeddings.csv", index=False)
print("Done! Embeddings written to completed_sa_all_trials_embeddings.csv and completed_sa_all_trials_chunk_embeddings.csv.")
