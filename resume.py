import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os 
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
from langchain.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import tempfile
import io
import time

# Define Pydantic models
class JDrequirements(BaseModel):
    """Answer the requirements in the form of Bulletpoints"""
    requirements: str = Field(description="Answer all the requirements in the form of bullet points, each starting with '*'")

class ComScore(BaseModel):
    """Measure the compatibility"""
    score: str = Field(description="Score the compatibility result between 0 to 100")
    requisite: str = Field(description="Requirements which are met or not")

class Recommender(BaseModel):
    """Give recommendation based on description"""
    recommendations: str = Field(description="A single string containing exactly 5 recommendations to enhance the compatibility score, each on a new line, starting with '*'")

class Synthesizer(BaseModel):
    """Gather the information in unified Format"""
    requirements: str = Field(description="Requirements of Job Description")
    score: str = Field(description="Score of Compatibility of job description and resume of Applicant")
    recommendation: str = Field(description="Recommendation for improvement of resume")

# Initialize the model
model = ChatGroq(model='llama-3.3-70b-versatile')

# Job Requirements Chain
llm_with_str = model.with_structured_output(JDrequirements)
system_jd = """You are an expert job requirement gathering assistant. Extract all essential requirements and skills from the provided job description. Return at least 5 requirements in bullet-point format, each starting with '*'. Ensure the output is a single string with each bullet point on a new line."""
prompt_jd = ChatPromptTemplate([
    ('system', system_jd),
    ('human', "Extract all requirements from the job description below:\n\n{description}")
])
JD_chain = prompt_jd | llm_with_str

# Compatibility Score Chain
llm_with_com = model.with_structured_output(ComScore)
system_comp = """You are an expert Analyzer which analyzes the compatibility between the job requirements and the applicant's resume. Score the compatibility between 0 and 100, where a higher score indicates better alignment. Provide a brief explanation of which requirements are met or not."""
prompt_comp = ChatPromptTemplate([
    ('system', system_comp),
    ('human', "Job requirements: {requirements}\nApplicant's resume: {resume}")
])
comp_chain = prompt_comp | llm_with_com

# Recommendation Chain
llm_with_rec = model.with_structured_output(Recommender)
system_rec = """You are an expert recommendation consultant with 20 years of experience. Based on the job requirements, applicant's resume, and compatibility score, provide exactly 5 recommendations to enhance the resume's compatibility with the job description. Return a single string with each recommendation starting with '*' and separated by new lines. Ensure exactly 5 bullet points are provided, each addressing a specific improvement to align the resume with the job requirements."""
prompt_rec = ChatPromptTemplate([
    ('system', system_rec),
    ('human', "Job description: {description}\nResume: {resume}\nCompatibility score: {score}")
])
recom_chain = prompt_rec | llm_with_rec

# Synthesizer Chain
llm_with_synth = model.with_structured_output(Synthesizer)
system_synth = """You are an expert Synthesizer. Format the provided information as follows, with each section titled:\n\n**Requirements**\n{requirements}\n\n**Score of Compatibility**\n{score}\n\n**Recommendation**\n{recommendation}"""
prompt_synth = ChatPromptTemplate([
    ('system', system_synth),
    ('human', "Requirements: {requirements}\nScore: {score}\nRecommendation: {recommendation}")
])
synthesizer_chain = prompt_synth | llm_with_synth

# Define State
class State(TypedDict):
    """State of Nodes of Graph"""
    job_desc: str
    resume: str
    requirements: str
    score: str
    recommendation: str
    finalized: Synthesizer

# Define Graph Nodes
def Job_req(state: State):
    """State the requirements mention in the job Description"""
    job_desc = state['job_desc']
    try:
        result = JD_chain.invoke({"description": job_desc})
        return {"requirements": result.requirements}
    except Exception as e:
        st.error(f"Error extracting job requirements: {str(e)}")
        return {"requirements": "* Unable to extract requirements due to an error."}

def compatability_Score(state: State):
    """Determine the compatibility score of Job Requirements and Resume of Applicant"""
    requirements = state["requirements"]
    resume = state['resume']
    try:
        result = comp_chain.invoke({"requirements": requirements, "resume": resume})
        return {'score': result.score}
    except Exception as e:
        st.error(f"Error calculating compatibility score: {str(e)}")
        return {'score': "0"}

def recom_node(state: State):
    """Give the recommendations based on the job requirements and applicant resume"""
    resume = state['resume']
    description = state["job_desc"]
    score = state['score']
    try:
        result = recom_chain.invoke({"description": description, "resume": resume, "score": score})
        return {"recommendation": result.recommendations}
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return {"recommendation": "* Unable to generate recommendations due to an error.\n* Please ensure the resume aligns with job requirements.\n* Consider adding relevant skills.\n* Review the job description for key qualifications.\n* Contact support for further assistance."}

def synthesizer(state: State):
    """Synthesize the document in proper format in the form of Markdown"""
    requirements = state["requirements"]
    recommendation = state["recommendation"]
    score = state['score']
    try:
        result = synthesizer_chain.invoke({"requirements": requirements, "score": score, "recommendation": recommendation})
        return {"finalized": result}
    except Exception as e:
        st.error(f"Error synthesizing results: {str(e)}")
        return {"finalized": Synthesizer(
            requirements="* Unable to synthesize requirements.",
            score="0",
            recommendation="* Unable to synthesize recommendations."
        )}

# Build the Graph
builder = StateGraph(State)
builder.add_node('job_req', Job_req)
builder.add_node('comp_score', compatability_Score)
builder.add_node('recommendation', recom_node)
builder.add_node('synthesizer', synthesizer)
builder.add_edge(START, 'job_req')
builder.add_edge('job_req', 'comp_score')
builder.add_edge('comp_score', 'recommendation')
builder.add_edge('recommendation', 'synthesizer')
builder.add_edge("synthesizer", END)
graph = builder.compile()

# Streamlit UI
st.set_page_config(page_title="Resume Analyzer", layout="wide")

st.title("Resume Analyzer")
st.markdown("Upload a job description and resume PDF to analyze compatibility and get recommendations.")

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Files")
    job_description_file = st.file_uploader("Upload Job Description (PDF)", type="pdf")
    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    analyze_button = st.button("Analyze")

# Main content area
if analyze_button and job_description_file and resume_file:
    with st.spinner("Analyzing..."):
        # Process job description PDF
        temp_job_file = None
        temp_resume_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_job_file:
                temp_job_file.write(job_description_file.read())
                temp_job_file_path = temp_job_file.name
                job_doc = PyPDFLoader(temp_job_file_path)
                job_docs = job_doc.load()
                job_description = job_docs[0].page_content if job_docs else ""

            # Process resume PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_resume_file:
                temp_resume_file.write(resume_file.read())
                temp_resume_file_path = temp_resume_file.name
                resume_doc = PyPDFLoader(temp_resume_file_path)
                resume_docs = resume_doc.load()
                resume = resume_docs[0].page_content if resume_docs else ""

            # Run the graph
            result = graph.invoke({"job_desc": job_description, "resume": resume})

            # Display results
            st.markdown("### ✅ Requirements")
            st.markdown(result['finalized'].requirements)

            st.markdown("### 📊 Score of Compatibility")
            st.markdown(result['finalized'].score)

            st.markdown("### 💡 Recommendations")
            st.markdown(result['finalized'].recommendation)

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
        finally:
            # Clean up temporary files with retry mechanism for Windows
            for temp_file_path in [temp_job_file.name if temp_job_file else None, temp_resume_file.name if temp_resume_file else None]:
                if temp_file_path and os.path.exists(temp_file_path):
                    for _ in range(5):  # Retry up to 5 times
                        try:
                            os.remove(temp_file_path)
                            break
                        except PermissionError:
                            time.sleep(0.1)  # Wait briefly before retrying
                        except Exception as e:
                            st.warning(f"Failed to delete temporary file {temp_file_path}: {str(e)}")
                            break
else:
    st.info("Please upload both a job description and a resume PDF, then click 'Analyze'.")