import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import tempfile
import shutil
import time
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

st.title("Job Description Matcher")

llm = ChatOpenAI(
    model_name="gpt-4o-mini-2024-07-18",
    max_tokens=15000,
    temperature=0.2,
)

prompt = ChatPromptTemplate.from_template(
"""
You are an expert in analyzing resumes with profound knowledge in technology, software engineering, data science, full stack web development, cloud engineering, 
cloud development, DevOps engineering, and big data engineering. 
Your role involves evaluating resumes against job descriptions.
Recognizing the competitive job market, provide top-notch assistance over the analysis of resumes against the job description.

1. **Keyword Match Percentage**:
   - **Semantic Similarity**: Focus on the meaning of technical words in the job description and resume, ensuring the use of embeddings to account for related technical terms only.
   - **Contextual Relevance**: Evaluate the relevance of matched technical keywords in the specific context of data engineering tasks, projects, or roles.
   - **Synonym Matching**: Strictly consider technical synonyms and related technical terms to ensure broader keyword coverage without considering common words.
   - **Keyword Density**: Analyze the frequency and distribution of important technical keywords across the resume.

2. **Skill Experience Match Percentage**:
   - **Experience Relevance**: Strictly evaluate the relevance of the candidate's experience to the specific technical skills, tasks, or projects mentioned in the job description.
   - **Skill Level Alignment**: Compare the required proficiency level for each technical skill in the job description with the candidate's proficiency.
   - **Recency of Experience**: Give higher weight to more recent experience with critical technical skills.
   - **Skill Utilization Frequency**: Consider the frequency and breadth of the candidate's use of each technical skill across different roles and projects.
   - **Project Complexity**: Assess the complexity of projects the candidate has worked on using specific technical skills.

3. **Cultural Fit Assessment**: Analyze how well the candidate's work experience and values align with the company's culture and the role's technical requirements.

4. **Educational Alignment**: Compare the candidate's educational background with the technical educational requirements or preferences in the job description.

5. **Certifications and Training**: Consider the relevance of any certifications or training programs that align with the technical requirements of the job description.

Ensure that only candidates who are highly relevant to the technical requirements of the job description are displayed. 
Also, provide a list of the candidate's strongest technical skills in order of relevance, and compare this order with the priority of technical skills in the job description. 

Today's date is 6th of August, 2024.

Job Description:
{input}

Resume Context:
{context}

The final response will strictly be in the following format:

Candidate 1: [Name]
Keyword Match: [Percentage]
//Calculation for above Percentage: Break down the percentage calculation by aggregating all 4, i.e., Semantic Similarity, Contextual Relevance, Synonym Matching and Keyword Density, and then provide the percentage. 
Skill Experience Match: [Percentage]
//Calculation for above Percentage: Break down the percentage calculation by aggregating all 4, i.e., Experience Relevance, Skill level alignment, recency of experience, skill utilization, and then provide the percentage.
Prominent Skills: [Highlight the most prominent technical skills for the respective job description in just one line.]
Overall Match: [Percentage]

Insight: [Provide a comprehensive analysis of each candidate's resume in relation to the specific job description. Highlight the candidate's key strengths, relevant technical experiences, and overall fit for the role. Include an evaluation of how well the candidate's technical skills, experiences, and accomplishments align with the job requirements, and suggest potential areas where the candidate may exceed or fall short of the desired qualifications.]
"""
)


# Create a temporary directory to store uploaded files
uploaded_files = st.file_uploader("Upload Resumes (PDF files)", type="pdf", accept_multiple_files=True)

def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    for file in uploaded_files:
        with open(os.path.join(temp_dir, file.name), "wb") as f:
            f.write(file.read())
    return temp_dir

def vector_embedding(temp_dir):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader(temp_dir)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

if uploaded_files:
    temp_dir = save_uploaded_files(uploaded_files)
    if st.button("Documents Embedding"):
        vector_embedding(temp_dir)
        st.write("Vector Store DB Is Ready")
        shutil.rmtree(temp_dir)  # Clean up temporary directory after embedding

job_description = st.text_area("Enter the Job Description", height=100)

if job_description:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': job_description})
    print("Response time:", time.process_time() - start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
