#!python -m pip install --upgrade pip
#!python -m venv .venv
pip install streamlit langchain langchain-openai langchain-chroma langchain-community langchain-google-genai langchain-huggingface chromadb sentence-transformers openai tiktoken python-dotenv

# Corporate Training RAG Bot with Streamlit
# Based on LangChain + ChromaDB + Gemini LLM
# Install: pip install streamlit langchain langchain-community langchain-google-genai chromadb sentence-transformers python-dotenv

import streamlit as st
from langchain_classic.chains import RetrievalQA, LLMChain
from langchain_classic.prompts import PromptTemplate
#from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
#from langchain_community.llms import OpenAI
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAI
import os
from datetime import datetime
# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Corporate Training RAG Bot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("RAG Chatbot with Streamlit")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left-color: #4caf50;
    }
    .info-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []
# SIMULATED CORPORATE TRAINING DATA
# ============================================================================

TRAINING_DOCUMENTS = [
    {
        "title": "Python Programming Fundamentals",
        "id": "TRN001",
        "category": "Technical Skills",
        "level": "Beginner",
        "duration": "40 hours",
        "content": """
        Python Programming Fundamentals (TRN001) is a comprehensive beginner-level course 
        designed for employees with little to no programming experience. This 40-hour training 
        covers essential Python concepts including data types, variables, control structures, 
        functions, and object-oriented programming basics. Participants will learn to write 
        clean, efficient Python code through hands-on exercises and real-world projects.
        
        Key topics include: Variables and data types, Conditional statements and loops, 
        Functions and modules, File handling, Basic OOP concepts, Error handling and debugging.
        
        Prerequisites: None. This course is suitable for complete beginners.
        Delivery: Online self-paced with weekly live Q&A sessions.
        Instructor: Dr. Sarah Johnson, Senior Python Developer with 10+ years experience.
        Certification: Upon completion, participants receive a Python Fundamentals Certificate.
        """
    },
    {
        "title": "Leadership and Management Excellence",
        "id": "TRN002",
        "category": "Soft Skills",
        "level": "Intermediate",
        "duration": "20 hours",
        "content": """
        Leadership and Management Excellence (TRN002) is an intermediate-level program 
        designed for current and aspiring managers. This 20-hour course develops essential 
        leadership skills including team management, conflict resolution, strategic thinking, 
        and effective communication. Participants learn proven leadership frameworks and 
        management techniques used by successful leaders across industries.
        
        Key topics include: Leadership styles and when to use them, Building high-performing teams,
        Conflict resolution strategies, Effective delegation, Performance management, 
        Strategic decision-making, Change management, Emotional intelligence in leadership.
        
        Prerequisites: Minimum 2 years of work experience, preferably in a supervisory role.
        Delivery: Hybrid format with in-person workshops and online modules.
        Instructor: Prof. Michael Chen, MBA, Executive Leadership Coach.
        Benefits: Improved team productivity, Better decision-making skills, Enhanced employee engagement.
        """
    },
    {
        "title": "Data Analysis with Excel and SQL",
        "id": "TRN003",
        "category": "Technical Skills",
        "level": "Intermediate",
        "duration": "30 hours",
        "content": """
        Data Analysis with Excel and SQL (TRN003) is an intermediate-level course that teaches 
        employees how to extract insights from data using industry-standard tools. This 30-hour 
        training covers advanced Excel functions, pivot tables, data visualization, SQL queries, 
        and database management. Participants will work with real datasets to solve business problems.
        
        Key topics include: Advanced Excel formulas and functions, Pivot tables and charts, 
        Data cleaning and preparation, SQL fundamentals (SELECT, JOIN, GROUP BY), 
        Database design basics, Creating dashboards, Statistical analysis in Excel.
        
        Prerequisites: Basic Excel knowledge (formulas, charts, basic functions).
        Delivery: Online instructor-led with hands-on lab exercises.
        Instructor: Jane Williams, Data Analytics Manager with 8 years experience.
        Tools used: Microsoft Excel 2019/365, MySQL or PostgreSQL.
        """
    },
    {
        "title": "Cybersecurity Awareness Training",
        "id": "TRN004",
        "category": "Compliance",
        "level": "Beginner",
        "duration": "8 hours",
        "content": """
        Cybersecurity Awareness Training (TRN004) is a mandatory beginner-level course for 
        all employees to protect company data and systems. This 8-hour training covers essential 
        security practices including password management, phishing detection, secure data handling, 
        and incident reporting. Employees learn to identify and respond to common cyber threats.
        
        Key topics include: Password security best practices, Identifying phishing emails and scams,
        Safe browsing and email habits, Data classification and protection, Mobile device security,
        Social engineering awareness, Incident reporting procedures, GDPR and data privacy basics.
        
        Prerequisites: None. Required for all employees.
        Delivery: Online self-paced with interactive scenarios and quizzes.
        Compliance: This training fulfills annual security awareness requirements.
        Certificate: Valid for 12 months, renewal required annually.
        """
    },
    {
        "title": "Machine Learning Fundamentals",
        "id": "TRN005",
        "category": "Technical Skills",
        "level": "Advanced",
        "duration": "50 hours",
        "content": """
        Machine Learning Fundamentals (TRN005) is an advanced-level course introducing employees 
        to ML algorithms and practical applications. This intensive 50-hour training covers 
        supervised and unsupervised learning, model evaluation, feature engineering, and deployment.
        Participants build ML models using Python and scikit-learn.
        
        Key topics include: Introduction to ML concepts and terminology, Supervised learning 
        (regression, classification), Unsupervised learning (clustering, dimensionality reduction),
        Model evaluation and validation, Feature engineering and selection, Hyperparameter tuning,
        Introduction to neural networks, ML model deployment basics.
        
        Prerequisites: Python programming experience, Basic statistics knowledge.
        Delivery: Online instructor-led with weekly assignments and a final project.
        Instructor: Dr. Robert Lee, Machine Learning Researcher, PhD in Computer Science.
        Tools: Python, Jupyter Notebooks, scikit-learn, pandas, numpy.
        """
    },
    {
        "title": "Effective Communication and Presentation Skills",
        "id": "TRN006",
        "category": "Soft Skills",
        "level": "Beginner",
        "duration": "16 hours",
        "content": """
        Effective Communication and Presentation Skills (TRN006) is a beginner-level course 
        designed to enhance verbal, written, and presentation abilities. This 16-hour training 
        teaches employees to communicate clearly, write professionally, deliver engaging presentations,
        and handle difficult conversations with confidence.
        
        Key topics include: Principles of effective communication, Active listening techniques,
        Professional email writing, Presentation structure and design, Public speaking skills,
        Body language and non-verbal communication, Handling Q&A sessions, Giving and receiving feedback.
        
        Prerequisites: None. Suitable for all employees.
        Delivery: In-person workshops with practice presentations.
        Instructor: Emma Davis, Professional Communication Coach, 12+ years experience.
        Benefits: Increased confidence, Better stakeholder relationships, Career advancement.
        """
    },
    {
        "title": "Agile and Scrum Methodology",
        "id": "TRN007",
        "category": "Project Management",
        "level": "Intermediate",
        "duration": "24 hours",
        "content": """
        Agile and Scrum Methodology (TRN007) is an intermediate-level course teaching modern 
        project management practices. This 24-hour training covers Agile principles, Scrum framework,
        sprint planning, daily standups, retrospectives, and tools for agile teams. Participants 
        learn to manage projects iteratively and deliver value faster.
        
        Key topics include: Agile manifesto and principles, Scrum roles (Product Owner, Scrum Master, Team),
        Sprint planning and execution, Daily standup meetings, Sprint review and retrospectives,
        User stories and backlog management, Agile estimation techniques, Kanban vs Scrum.
        
        Prerequisites: Basic project management experience helpful but not required.
        Delivery: Hybrid with hands-on Scrum simulations.
        Instructor: David Kumar, Certified Scrum Master (CSM), Agile Coach.
        Certification: Prepares for Certified ScrumMaster (CSM) exam.
        """
    },
    {
        "title": "Financial Planning and Analysis",
        "id": "TRN008",
        "category": "Finance",
        "level": "Advanced",
        "duration": "35 hours",
        "content": """
        Financial Planning and Analysis (TRN008) is an advanced-level course for finance professionals.
        This comprehensive 35-hour training covers financial modeling, budgeting, forecasting, 
        variance analysis, and strategic financial planning. Participants learn to create complex 
        financial models and provide data-driven insights to senior management.
        
        Key topics include: Financial statement analysis, Building financial models in Excel,
        Budgeting processes and best practices, Financial forecasting techniques, Variance analysis,
        Cash flow management, Investment appraisal, Financial ratios and KPIs, Scenario analysis.
        
        Prerequisites: Accounting fundamentals, Advanced Excel skills.
        Delivery: Online instructor-led with case studies from real companies.
        Instructor: CFA John Smith, Director of FP&A with 15 years experience.
        Target audience: Finance analysts, FP&A professionals, Controllers.
        """
    }
]
# Initialize RAG components (this is a simplified example)
@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    embeddings = OpenAIEmbeddings()
    # Replace with your vector store and retriever logic
    vectorstore = Chroma(
        embedding_function=embeddings, 
        collection_name="my_rag_collection"
    )
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(model="gemini-2.5-flash",api_key= OPENAI_API_KEY,model_provider="google_genai"),
        chain_type="stuff",
        retriever=retriever
    )
    return qa_chain

rag_chain = load_rag_pipeline()
#Step 3: Add document handling
import tempfile

def handle_file_upload():
    uploaded_file = st.sidebar.file_uploader("Upload a document", type=["txt", "pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Here you would add your logic to load, split, and index the document
        # with LangChain or a similar library.
        st.success("Document uploaded and processed!")
        os.remove(tmp_file_path) # Clean up the temporary file
    
handle_file_upload()
#Step 4: Display chat history and handle user input
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Call the RAG chain to get a response
        with st.spinner("Thinking..."):
            response = rag_chain.run(prompt)
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})


#Step 5: Run the application
!streamlit run app.py


