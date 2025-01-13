import streamlit as st
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from groq import Groq
import json
import fitz
import matplotlib.pyplot as plt
import seaborn as sns

# Set device for BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

###################### Start #######################
# Llama 3.1 Initialization
client = Groq(api_key="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")

# Adjusted prompt for JSON output
instruction = """
You are an AI bot designed to parse resumes and extract the following details in below JSON:
1. full_name: 
2. university_name: of mosrt recent degree (return the short form of the university name else return full name) 
3. national_university/international_university: "return National if inside Pak else return International" Like 'national_university/international_university': 'National'
4. email_id: if available else return "N/A"
5. github_link: if available else return "N/A"
6. employment_details: (with fields: company, position, return total years of experience, location, and tags indicating teaching/industry/internship) Like employment_details': [{'company': '11values PVT Ltd', 'position': 'Senior Software Engineer', 'years_of_experience': '1.1 years', 'location': 'Rawalpindi, Pakistan', 'tags': 'industry'}]
7. total_professional_experience: total experience in years like 2.5 years excluding internships if not available return Fresh Graduate
8. technical_skills: Like 'technical_skills': ['Python', 'Machine Learning', 'Deep Learning']
9. soft_skills: 
10. location: Like 'location': 'Karachi, Pakistan'

Return all information in JSON format.
"""
###################### End #######################

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# Function to get BERT embeddings for a text input
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().cpu()

# Function to safely decode the file content with fallback encoding
def decode_file(file):
    try:
        return file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        return file.getvalue().decode("ISO-8859-1")

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Custom CSS Styling for the Streamlit App
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: white;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .streamlit-expanderHeader {
        font-size: 18px;
        font-weight: bold;
        color: #3498db;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stDataFrame {
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    .stSidebar {
        background-color: #2c3e50;
        color: white;
    }
    .stMarkdown {
        margin-top: 20px;
    }
    table, th, td {
        border: 1px solid #ccc;
    }
    th, td {
        padding: 8px 12px;
    }
    .stAlert {
        background-color: #2ecc71;
        color: white;
        padding: 10px;
        font-weight: bold;
    }
    /* Custom CSS for file uploader label */
    label {
        color: white !important;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)


# Set up Streamlit app UI
st.title("Automated Resume Screening Dashboard")

# Sidebar - File Upload
with st.sidebar:
    st.header("📤 File Uploads")
    jd_file = st.file_uploader("Upload Job Description (.txt or .pdf)", type=["txt", "pdf"])
    resume_files = st.file_uploader("Upload Resumes (.txt or .pdf)", type=["txt", "pdf"], accept_multiple_files=True)

########################## Main Body ###########################

# Only process if JD and resumes are uploaded
if jd_file and resume_files:
    # Initialize results_df only after resumes are processed
    results_df = pd.DataFrame(columns=["Resume", "Similarity Score", "full_name", "university_name","national/international  uni.","email_id","github_link", "company_names", "technical_skills", "soft_skills", "Total experience in Years"])

    # Process files and calculate similarity only if resumes are uploaded
    if jd_file and resume_files:
        if jd_file.type == "application/pdf":
            jd_content = extract_text_from_pdf(jd_file)
        else:
            jd_content = decode_file(jd_file)
        jd_content = preprocess_text(jd_content)
        jd_embedding = get_bert_embeddings(jd_content)

        results = []
        for resume_file in resume_files:
            if resume_file.type == "application/pdf":
                resume_content = extract_text_from_pdf(resume_file)
            else:
                resume_content = decode_file(resume_file)
            resume_content = preprocess_text(resume_content)
            resume_embedding = get_bert_embeddings(resume_content)
            similarity_score = cosine_similarity(jd_embedding, resume_embedding)[0][0]

            # Request data extraction from Groq
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": instruction + resume_content}],
                temperature=0, max_tokens=1024, top_p=0.65, response_format={"type": "json_object"}
            )

            try:
                result_json = completion.choices[0].message.content
                result = json.loads(result_json)
            except json.JSONDecodeError:
                result = {}

            print(result)

            employment_details = result.get("employment_details", [])
            
            # Store employment details and classify
            if not employment_details:
                company_names = ["N/A"]
            else:
                company_names = [detail.get('company', 'N/A') for detail in employment_details]
            
            # Append data to results
            results.append({
                'full_name': result.get("full_name"),
                'Similarity Score': similarity_score,
                'university_name': result.get("university_name"),
                "national/international  uni." : result.get("national_university/international_university"),
                "email_id": result.get("email_id", "N/A"),
                "github_link": result.get("github_link", "N/A"),
                'company_names': company_names,
                'technical_skills': result.get("technical_skills", []),
                'soft_skills': result.get("soft_skills", []),
                "Total experience in Years": result.get("total_professional_experience"),
                'location': result.get("location")
            })

        # Create DataFrame from results
        results_df = pd.DataFrame(results)

    # Sort results by Similarity Score in descending order
    results_df = results_df.sort_values(by="Similarity Score", ascending=False)
    # Display filtered table
    st.write("### Candidates")
    st.dataframe(results_df)

    ######################### Filter Section ######################
    # Multi-select filters for university, company, and skills
    st.write("### Apply Filters")

    # Filters for universities and companies
    universities = st.multiselect("Select Universities", options=results_df["university_name"].unique())
    companies = st.multiselect("Select Companies", options=results_df["company_names"].explode().unique())
    skills = st.multiselect("Select Skills", options=results_df['technical_skills'].explode().unique())

    # Filter results based on selections
    filtered_df = results_df.copy()

    if universities:
        filtered_df = filtered_df[filtered_df["university_name"].isin(universities)]
    if companies:
        filtered_df = filtered_df[filtered_df['company_names'].apply(lambda x: any(company in x for company in companies))]
    if skills:
        filtered_df = filtered_df[filtered_df['technical_skills'].apply(lambda x: any(skill in x for skill in skills))]

    st.write("### Filtered Candidates")
    st.dataframe(filtered_df)

    ######################### Resume Statistics Table ######################
    # Experience and university/company counts
    flattened_company_names = [company for sublist in filtered_df['company_names'] for company in sublist]
    unique_companies = list(set(flattened_company_names))

    
    ######################### Pie Chart for Skills ######################
    skill_counts = filtered_df['technical_skills'].explode().value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(skill_counts, labels=skill_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.write("### Skill Distribution")
    st.pyplot(fig)
