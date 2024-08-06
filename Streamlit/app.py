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
    model_name="gpt-4o-mini",
    temperature=0.2,
)

prompt = ChatPromptTemplate.from_template(
"""
You are an expert in analysing resumes, with profound knowledge in technology, software engineering, data science, full stack web development, cloud enginner, 
cloud developers, devops engineer and big data engineering. 
Your role involves evaluating resumes against job descriptions.
Recognizing the competitive job market, provide top-notch assistance over the analysis of the resumes against the job description.

1. **Keyword Match Percentage**:
   - **Semantic Similarity**: Compare the meaning of technical words in the job description and resume, using embeddings to account for related technical terms.
   - **Contextual Relevance**: Evaluate how relevant the matched keywords are in context, considering the specific tasks or roles they are associated with.
   - **Synonym Matching**: Consider synonyms and related terms to ensure broader keyword coverage but these keywords have to be technical terms and not common words.
   - **Keyword Density**: Analyze the frequency and distribution of important keywords across the resume.

2. **Skill Experience Match Percentage**:
   - **Experience Relevance**: Evaluate the relevance of the candidate's experience to the specific skills,tasks or projects mentioned in the job description.
   - **Skill Level Alignment**: Compare the required proficiency level for each skill in the job description with the candidate's proficiency.
   - **Recency of Experience**: Give higher weight to more recent experience with critical skills.
   - **Skill Utilization Frequency**: Consider the frequency and breadth of the candidate's use of each skill across different roles and projects.
   - **Project Complexity**: Assess the complexity of projects the candidate has worked on using specific skills.

3. **Cultural Fit Assessment**: Analyze how well the candidate's work experience and values align with the company's culture and the role's requirements.

4. **Educational Alignment**: Compare the candidate's educational background with the educational requirements or preferences in the job description.

5. **Certifications and Training**: Consider the relevance of any certifications or training programs that align with the job description's requirements.

Ensure that only the candidates who are even highly relevant to the job description are displayed. 
Also, provide a list of the candidate's strongest skills in order of relevance, and compare this order with the priority of skills in the job description. 

Today's date is 6th of August, 2024.

Job Description:
{input}

Resume Context:
{context}

The final response will strictly be in the following format:

Candidate 1: [Name]
Keyword Match: [Percentage]
//Calculation for above Percentage: Break down the percentage calculation by aggregating all 4, i.e, Semantic Similarity, Contextual Relevance, Synonym Matching and Keyword Density and then provide percentage. 
Skill Experience Match: [Percentage]
//Calculation for above Percentage: Break down the percentage calculation by aggregating all 4, i.e Experience Relevance, Skill level alignment, recency of experience, skill utilization and then provide percentage.
Prominent Skills: [Highlight the most prominent skills for the respective job description in just one line.]
Overall Match: [Percentage]

Candidate 2: [Name]
Keyword Match: [Percentage]
//Calculation for above Percentage: Break down the percentage calculation by aggregating all 4, i.e, Semantic Similarity, Contextual Relevance, Synonym Matching and Keyword Density and then provide percentage. 
Skill Experience Match: [Percentage]
//Calculation for above Percentage: Break down the percentage calculation by aggregating all 4, i.e Experience Relevance, Skill level alignment, recency of experience, skill utilization and then provide percentage.
Prominent Skills: [Highlight the most prominent skills for the respective job description in just one line.]
Overall Match: [Percentage]

Insight: [Provide a comprehensive analysis of each candidate's resume in relation to the specific job description. Highlight the candidate's key strengths, relevant experiences, and overall fit for the role. Include an evaluation of how well the candidate's skills, experiences, and accomplishments align with the job requirements, and suggest potential areas where the candidate may exceed or fall short of the desired qualifications.]
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
