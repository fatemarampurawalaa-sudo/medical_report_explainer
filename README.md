# medical_report_explainer
AI Medical Report Explainer – RAG chatbot that explains lab reports in simple English (demo data only)
# 🏥 AI Medical Report Explainer

An AI chatbot that explains blood test reports in simple English, using **RAG (Retrieval-Augmented Generation)**.

> ⚠️ **Demo project only.** Uses a sample report with made-up data. Not medical advice.

## What it does
- Reads a lab report PDF and splits it into sections (Lipids, Diabetes, Vitamins, etc.)
- Finds the right section for each question using embeddings + a vector database
- Explains each value: the result, the normal range, and whether it is normal, high or low
- Shows which section the answer came from
- Says **"I don't know"** when the answer isn't in the report

## Safety rules
- Never diagnoses a disease or suggests medicine doses
- Always ends with "Please talk to your doctor"
- Treats report text as data, not instructions (prompt-injection protection)

## Tech stack
Python · Gemini API · Embeddings · ChromaDB · RAG · Streamlit

## How it works
1. **Chunk**: split the report by section headings
2. **Embed**: turn each section into numbers that capture its meaning
3. **Search**: find the sections closest to the question
4. **Answer**: Gemini answers using only those sections, with safety rules

## Challenge solved
Chunking split the report in the wrong place because "PREDIABETES" contains "DIABETES".
Fixed by matching section headings only at the start of a line.

## Tests
Checked for correct values, "I don't know" on missing data, no medicine doses, and prompt-injection refusal.
