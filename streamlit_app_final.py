# ================== IMPORTS ==================
import os
import time
import re
import numpy as np
import streamlit as st
import pdfplumber
from docx import Document
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import qrcode
from PIL import Image
from pymongo import MongoClient
import uuid
from datetime import datetime, timedelta
from openai import OpenAI
client = OpenAI()


MONGO_URI = "mongodb+srv://teena3:123@cluster0.ojomaf6.mongodb.net/" # connection string 

# Connect to MongoDB
client = MongoClient(MONGO_URI)

# Database aur Collection choose karo
db = client["myDatabase"]        
collection = db["user_data"]     # main user collection
resume_collection = db["resume_analysis"]  # resume analysis data ke liye
jd_collection = db["jd_matching"]          # JD matching data ke liye
payment_collection = db["payments"]        # payment history ke liye
pending_payments = db["pending_payments"]  # pending payments track karne ke liye

hr_collection = db["hr_users"]              # HR registration data
hr_shortlist_collection = db["hr_shortlists"]  # HR shortlisted candidates

# ================== POSITION CLASSIFICATION ==================
def detect_best_position(resume_text):
    """Resume text ke basis pe best suited position detect karta hai"""
    
    # Position keywords mapping
    position_keywords = {
        "Data Analyst": [
            "data analysis", "sql", "python", "excel", "tableau", "power bi", 
            "statistics", "data visualization", "pandas", "numpy", "analytics",
            "business intelligence", "reporting", "dashboard", "kpi", "metrics"
        ],
        "Data Scientist": [
            "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn",
            "nlp", "computer vision", "neural network", "predictive modeling", "ai",
            "data science", "feature engineering", "model deployment", "r programming"
        ],
        "Software Engineer": [
            "software development", "programming", "java", "c++", "javascript", "react",
            "angular", "node.js", "api", "microservices", "agile", "git", "devops",
            "backend", "frontend", "full stack", "coding", "algorithm"
        ],
        "Project Manager": [
            "project management", "agile", "scrum", "stakeholder", "budget", "timeline",
            "risk management", "team leadership", "pmp", "jira", "coordination",
            "planning", "execution", "delivery", "roadmap", "sprint"
        ],
        "Business Analyst": [
            "business analysis", "requirements gathering", "stakeholder management",
            "process improvement", "documentation", "user stories", "wireframes",
            "gap analysis", "feasibility study", "business requirements", "uml"
        ],
        "DevOps Engineer": [
            "devops", "ci/cd", "jenkins", "docker", "kubernetes", "aws", "azure",
            "terraform", "ansible", "monitoring", "automation", "cloud", "pipeline",
            "deployment", "infrastructure", "containerization"
        ],
        "Frontend Developer": [
            "html", "css", "javascript", "react", "angular", "vue", "ui/ux",
            "responsive design", "bootstrap", "sass", "webpack", "jquery",
            "frontend development", "web development", "dom manipulation"
        ],
        "Backend Developer": [
            "backend development", "api development", "database", "server", "node.js",
            "django", "flask", "spring boot", "rest api", "graphql", "mongodb",
            "postgresql", "redis", "microservices", "authentication"
        ],
        "Marketing Manager": [
            "marketing strategy", "digital marketing", "seo", "sem", "content marketing",
            "social media", "brand management", "campaign", "analytics", "roi",
            "market research", "email marketing", "advertising", "growth hacking"
        ],
        "HR Manager": [
            "human resources", "recruitment", "talent acquisition", "employee relations",
            "performance management", "onboarding", "training", "compensation",
            "hr policies", "workforce planning", "hrms", "payroll"
        ],
        "Sales Executive": [
            "sales", "business development", "lead generation", "client relationship",
            "negotiation", "revenue", "crm", "salesforce", "cold calling",
            "account management", "sales strategy", "targets", "pipeline"
        ],
        "UI/UX Designer": [
            "ui design", "ux design", "figma", "sketch", "adobe xd", "wireframing",
            "prototyping", "user research", "usability testing", "design thinking",
            "interaction design", "user interface", "user experience"
        ],
        "Product Manager": [
            "product management", "product strategy", "roadmap", "feature prioritization",
            "user stories", "product lifecycle", "market research", "mvp",
            "product analytics", "a/b testing", "stakeholder management"
        ],
        "QA Engineer": [
            "quality assurance", "testing", "automation testing", "selenium", "junit",
            "test cases", "bug tracking", "regression testing", "manual testing",
            "test automation", "qa processes", "defect management"
        ],
        "Content Writer": [
            "content writing", "copywriting", "seo writing", "blog", "article",
            "creative writing", "editing", "proofreading", "content strategy",
            "storytelling", "content management", "cms", "wordpress"
        ]
    }
    
    resume_lower = resume_text.lower()
    position_scores = {}
    
    # Calculate score for each position
    for position, keywords in position_keywords.items():
        score = 0
        matched_keywords = []
        
        for keyword in keywords:
            if keyword.lower() in resume_lower:
                score += 1
                matched_keywords.append(keyword)
        
        # Calculate percentage match
        match_percentage = (score / len(keywords)) * 100
        position_scores[position] = {
            "score": score,
            "percentage": match_percentage,
            "matched_keywords": matched_keywords
        }
    
    # Sort positions by score
    sorted_positions = sorted(position_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    
    # Get top 3 positions
    best_positions = []
    for position, data in sorted_positions[:3]:
        if data["score"] > 0:  # Only include if there's at least some match
            best_positions.append({
                "position": position,
                "match_percentage": round(data["percentage"], 1),
                "keyword_count": data["score"],
                "confidence": "High" if data["percentage"] >= 30 else "Medium" if data["percentage"] >= 15 else "Low"
            })
    
    return best_positions if best_positions else [{"position": "General", "match_percentage": 0, "keyword_count": 0, "confidence": "Low"}]
# ================== HR HELPER FUNCTIONS ==================

def save_hr_registration(company_name, hr_name, hr_email, phone, company_size, industry):
    """HR registration data save karta hai"""
    try:
        hr_data = {
            "company_name": company_name,
            "hr_name": hr_name,
            "hr_email": hr_email,
            "phone": phone,
            "company_size": company_size,
            "industry": industry,
            "registration_date": datetime.now(),
            "status": "active"
        }
        
        # Check if already exists
        existing = hr_collection.find_one({"hr_email": hr_email})
        if existing:
            return False, "Email already registered!"
        
        hr_collection.insert_one(hr_data)
        return True, "Registration successful!"
    except Exception as e:
        return False, f"Error: {str(e)}"

def verify_hr_login(email, company_name):
    """HR login verify karta hai"""
    try:
        hr_user = hr_collection.find_one({
            "hr_email": email,
            "company_name": company_name,
            "status": "active"
        })
        
        if hr_user:
            return True, hr_user
        return False, None
    except Exception as e:
        return False, None

def get_filtered_resumes(position=None, min_ats_score=0):
    """Filtered resumes return karta hai"""
    try:
        query = {}
        
        # Build query based on filters
        if position and position != "All Positions":
            query["best_position.position"] = position
        
        if min_ats_score > 0:
            query["ats_score"] = {"$gte": min_ats_score}
        
        # Get all resumes matching query
        resumes = list(resume_collection.find(query).sort("ats_score", -1).limit(50))
        
        return resumes
    except Exception as e:
        st.error(f"Error fetching resumes: {str(e)}")
        return []

def shortlist_candidate(hr_email, student_email, resume_id):
    """Candidate ko shortlist karta hai"""
    try:
        # Check if already shortlisted
        existing = hr_shortlist_collection.find_one({
            "hr_email": hr_email,
            "student_email": student_email
        })
        
        if existing:
            return False
        
        shortlist_data = {
            "hr_email": hr_email,
            "student_email": student_email,
            "resume_id": resume_id,
            "shortlisted_date": datetime.now(),
            "status": "shortlisted"
        }
        
        hr_shortlist_collection.insert_one(shortlist_data)
        return True
    except:
        return False

def get_shortlisted_candidates(hr_email):
    """HR ke shortlisted candidates return karta hai"""
    try:
        shortlisted = list(hr_shortlist_collection.find({"hr_email": hr_email}))
        return shortlisted
    except:
        return []

# ================== HELPER FUNCTIONS ==================
def generate_qr(url):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    return img

def generate_payment_link(user_email, amount):
    payment_id = str(uuid.uuid4())[:8]
    payment_url = f"https://payment.example.com?id={payment_id}&email={user_email}&amount={amount}"
    
    pending_data = {
        "payment_id": payment_id,
        "user_email": user_email,
        "amount": amount,
        "status": "pending",
        "created_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(hours=24)
    }
    pending_payments.insert_one(pending_data)
    
    return payment_url, payment_id

def verify_payment(payment_id):
    payment = pending_payments.find_one({"payment_id": payment_id})
    if payment and payment["status"] == "pending":
        current_time = datetime.now()
        if current_time < payment["expires_at"]:
            pending_payments.update_one(
                {"payment_id": payment_id},
                {"$set": {"status": "completed", "completed_at": current_time}}
            )
            
            payment_record = {
                "payment_id": payment_id,
                "user_email": payment["user_email"],
                "amount": payment["amount"],
                "payment_date": current_time,
                "status": "success"
            }
            payment_collection.insert_one(payment_record)
            
            collection.update_one(
                {"email": payment["user_email"]},
                {"$set": {"payment_status": "completed", "payment_date": current_time}}
            )
            
            return True
    return False

def has_active_payment(email):
    user = collection.find_one({"email": email})
    if user and user.get("payment_status") == "completed":
        return True
    return False

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
        return ""

def extract_text_from_docx(docx_file):
    try:
        doc = Document(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"DOCX extraction error: {str(e)}")
        return ""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s@.\-]', '', text)
    return text.strip()

def calculate_ats_score(resume_text, jd_text):
    """Enhanced ATS score calculation with better accuracy"""
    
    # Clean texts
    resume_clean = clean_text(resume_text.lower())
    jd_clean = clean_text(jd_text.lower())
    
    # TF-IDF based similarity
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    try:
        vectors = vectorizer.fit_transform([resume_clean, jd_clean])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        base_score = similarity * 100
    except:
        base_score = 0
    
    # Keyword matching bonus
    jd_keywords = set(jd_clean.split())
    resume_keywords = set(resume_clean.split())
    
    if len(jd_keywords) > 0:
        keyword_match_ratio = len(jd_keywords.intersection(resume_keywords)) / len(jd_keywords)
        keyword_bonus = keyword_match_ratio * 20
    else:
        keyword_bonus = 0
    
    # Technical terms bonus
    tech_terms = ['python', 'java', 'javascript', 'sql', 'aws', 'docker', 'kubernetes', 
                  'machine learning', 'data analysis', 'react', 'angular', 'node.js']
    
    jd_has_tech = any(term in jd_clean for term in tech_terms)
    resume_has_tech = any(term in resume_clean for term in tech_terms)
    
    tech_bonus = 10 if (jd_has_tech and resume_has_tech) else 0
    
    # Final score calculation
    final_score = min(base_score + keyword_bonus + tech_bonus, 100)
    
    return round(final_score, 2)

def analyze_resume_with_ai(resume_text):
    """AI se resume analysis karta hai using GPT"""
    try:
        prompt = f"""
        Analyze this resume and provide:
        1. Key strengths (3-4 points)
        2. Areas for improvement (3-4 points)
        3. Overall impression (2-3 sentences)
        
        Resume:
        {resume_text[:2000]}
        
        Provide concise, actionable feedback.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert resume reviewer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

def get_resume_suggestions(resume_text, jd_text, ats_score):
    """Resume improvement suggestions provide karta hai"""
    try:
        prompt = f"""
        Based on this resume and job description, provide 5 specific suggestions to improve ATS score.
        Current ATS Score: {ats_score}%
        
        Job Description keywords: {jd_text[:500]}
        Resume: {resume_text[:500]}
        
        Provide actionable suggestions.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an ATS optimization expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )
        
        return response.choices[0].message.content
    except:
        return "Suggestions unavailable at the moment."

def save_resume_analysis(user_email, resume_text, ats_score, best_position):
    """Resume analysis ko database mein save karta hai"""
    try:
        analysis_data = {
            "user_email": user_email,
            "resume_text": resume_text,
            "ats_score": ats_score,
            "best_position": best_position,
            "analysis_date": datetime.now(),
            "status": "analyzed"
        }
        
        # Update if exists, insert if new
        resume_collection.update_one(
            {"user_email": user_email},
            {"$set": analysis_data},
            upsert=True
        )
        return True
    except Exception as e:
        st.error(f"Error saving analysis: {str(e)}")
        return False

# ================== MAIN APP ==================

def main():
    st.set_page_config(
        page_title="AI Resume Platform",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS - IMPROVED STYLING
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .portal-card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
            margin: 1rem;
            min-height: 200px;
        }
        .portal-card h3 {
            color: white;
            font-size: 1.8rem;
            margin-bottom: 1rem;
        }
        .portal-card p {
            color: white;
            font-size: 1.1rem;
            line-height: 1.6;
        }
        .portal-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }
        .stButton > button {
            width: 100%;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            font-weight: bold;
            border-radius: 10px;
            transition: all 0.3s;
        }
        .feature-box {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        .student-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .hr-section {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        .info-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .info-card h3 {
            color: #333;
            margin-bottom: 1rem;
        }
        .info-card ul {
            list-style: none;
            padding: 0;
        }
        .info-card li {
            padding: 0.5rem 0;
            color: #555;
            font-size: 1rem;
        }
        .centered-container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
        }
        .resume-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .position-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            margin: 5px;
            font-weight: bold;
            font-size: 0.9rem;
        }
        .badge-green {
            background: #4CAF50;
            color: white;
        }
        .badge-orange {
            background: #FF9800;
            color: white;
        }
        .badge-blue {
            background: #2196F3;
            color: white;
        }
        .stats-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session states
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'hr_logged_in' not in st.session_state:
        st.session_state.hr_logged_in = False
    if 'hr_email' not in st.session_state:
        st.session_state.hr_email = None
    
    # ================== HOME PAGE ==================
    if st.session_state.page == 'home':
        st.markdown("""
        <div class="main-header">
            <h1>🎯 AI-Powered Resume Platform</h1>
            <p style="font-size: 1.2rem;">Smart Resume Analysis & Recruitment Solution</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Portal Selection - SIDE BY SIDE
        st.markdown("<h2 style='text-align: center; margin: 3rem 0 2rem 0;'>Choose Your Portal</h2>", unsafe_allow_html=True)
        
        # Create two columns for side-by-side layout
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("""
            <div class="portal-card student-section">
                <h3>🎓 Student Portal</h3>
                <p>Upload your resume, get AI-powered analysis, and improve your ATS score</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📚 Enter Student Portal", key="student_btn"):
                st.session_state.page = 'student_login'
                st.rerun()
        
        with col_right:
            st.markdown("""
            <div class="portal-card hr-section">
                <h3>💼 HR Portal</h3>
                <p>Search candidates, view resumes, and build your talent pipeline</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("💼 Enter HR Portal", key="hr_btn"):
                st.session_state.page = 'hr_login'
                st.rerun()
        
        # Features Section
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center; margin: 3rem 0 2rem 0;'>✨ Platform Features</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h3>🎓 For Students</h3>
                <ul>
                    <li>✅ AI-Powered Resume Analysis</li>
                    <li>✅ ATS Score Calculation</li>
                    <li>✅ Position Recommendations</li>
                    <li>✅ Improvement Suggestions</li>
                    <li>✅ Job Description Matching</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h3>💼 For HR Professionals</h3>
                <ul>
                    <li>✅ Advanced Candidate Search</li>
                    <li>✅ Filter by Skills & ATS Score</li>
                    <li>✅ Resume Database Access</li>
                    <li>✅ Candidate Shortlisting</li>
                    <li>✅ Contact Information Access</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # ================== STUDENT LOGIN PAGE ==================
    elif st.session_state.page == 'student_login':
        # Back button
        if st.button("← Back to Home"):
            st.session_state.page = 'home'
            st.rerun()
        
        st.markdown("""
        <div class="main-header">
            <h1>🎓 Student Portal</h1>
            <p>Login or Register to Continue</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Centered login form
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
            
            with tab1:
                st.markdown("<br>", unsafe_allow_html=True)
                email = st.text_input("Email Address", key="login_email")
                name = st.text_input("Name", key="login_name")
                
                if st.button("Login", type="primary"):
                    if email and name:
                        user = collection.find_one({"email": email, "name": name})
                        if user:
                            st.session_state.logged_in = True
                            st.session_state.user_email = email
                            st.session_state.page = 'student_dashboard'
                            st.success("Login successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Invalid credentials! Please register first.")
                    else:
                        st.warning("Please fill all fields")
            
            with tab2:
                st.markdown("<br>", unsafe_allow_html=True)
                reg_name = st.text_input("Full Name", key="reg_name")
                reg_email = st.text_input("Email Address", key="reg_email")
                reg_phone = st.text_input("Phone Number", key="reg_phone")
                reg_college = st.text_input("College/University", key="reg_college")
                
                if st.button("Register", type="primary"):
                    if reg_name and reg_email and reg_phone:
                        existing_user = collection.find_one({"email": reg_email})
                        if existing_user:
                            st.error("Email already registered! Please login.")
                        else:
                            user_data = {
                                "name": reg_name,
                                "email": reg_email,
                                "phone": reg_phone,
                                "college": reg_college,
                                "registration_date": datetime.now(),
                                "payment_status": "pending"
                            }
                            collection.insert_one(user_data)
                            st.success("Registration successful! Please login.")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.warning("Please fill all required fields")
    
    # ================== STUDENT DASHBOARD ==================
    elif st.session_state.page == 'student_dashboard':
        if not st.session_state.logged_in:
            st.session_state.page = 'student_login'
            st.rerun()
        
        # Logout button
        col1, col2, col3 = st.columns([6, 1, 1])
        with col3:
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.user_email = None
                st.session_state.page = 'home'
                st.rerun()
        
        # Get user details
        user = collection.find_one({"email": st.session_state.user_email})
        
        st.markdown(f"""
        <div class="main-header">
            <h1>Welcome, {user.get('name', 'Student')}! 🎓</h1>
            <p>Your AI-Powered Resume Analysis Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Payment check
        if not has_active_payment(st.session_state.user_email):
            st.warning("⚠️ Payment required to access full features")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div class="info-card">
                    <h3>💳 Complete Payment</h3>
                    <p><strong>Amount:</strong> ₹99 (One-time)</p>
                    <p><strong>Features Unlocked:</strong></p>
                    <ul>
                        <li>✅ Unlimited Resume Analysis</li>
                        <li>✅ AI-Powered Feedback</li>
                        <li>✅ ATS Score Tracking</li>
                        <li>✅ Job Matching</li>
                        <li>✅ Masterclass Access</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Generate Payment Link", type="primary"):
                    payment_url, payment_id = generate_payment_link(st.session_state.user_email, 99)
                    
                    st.success(f"Payment link generated! Payment ID: {payment_id}")
                    
                    qr_img = generate_qr(payment_url)
                    
                    col_qr1, col_qr2, col_qr3 = st.columns([1, 2, 1])
                    with col_qr2:
                        st.image(qr_img, caption="Scan to Pay")
                        st.code(payment_url, language="text")
                    
                    st.info("👆 Scan the QR code or use the link above to complete payment")
                    
                    verify_payment_id = st.text_input("Enter Payment ID to verify:", key="verify_payment")
                    if st.button("Verify Payment"):
                        if verify_payment(verify_payment_id):
                            st.success("✅ Payment verified! Refreshing...")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Payment not found or expired")
        else:
            # Main Dashboard - FULL TABS
            st.success("✅ Payment Status: Active")
            
            # Dashboard tabs - ALL 5 TABS
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📄 Resume Analysis", 
                "🎯 JD Matching", 
                "🎓 Masterclass",
                "ℹ️ About",
                "👤 Profile"
            ])
            
            # TAB 1: Resume Analysis
            with tab1:
                st.markdown("### 📄 Upload & Analyze Resume")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    uploaded_file = st.file_uploader(
                        "Upload your resume (PDF or DOCX)",
                        type=['pdf', 'docx'],
                        key="resume_upload"
                    )
                    
                    if uploaded_file:
                        st.info(f"📎 File uploaded: {uploaded_file.name}")
                        
                        # Extract text
                        if uploaded_file.type == "application/pdf":
                            resume_text = extract_text_from_pdf(uploaded_file)
                        else:
                            resume_text = extract_text_from_docx(uploaded_file)
                        
                        if resume_text:
                            # Detect best position
                            best_positions = detect_best_position(resume_text)
                            
                            st.markdown("### 🎯 Recommended Positions")
                            
                            for idx, pos in enumerate(best_positions[:3]):
                                confidence_color = "badge-green" if pos['confidence'] == 'High' else "badge-orange" if pos['confidence'] == 'Medium' else "badge-blue"
                                
                                st.markdown(f"""
                                <div class="resume-card">
                                    <h4>#{idx+1} {pos['position']}</h4>
                                    <span class="position-badge {confidence_color}">
                                        {pos['confidence']} Confidence
                                    </span>
                                    <span class="position-badge badge-blue">
                                        {pos['match_percentage']}% Match
                                    </span>
                                    <span class="position-badge badge-orange">
                                        {pos['keyword_count']} Matching Skills
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # AI Analysis
                            if st.button("🤖 Get AI Analysis"):
                                with st.spinner("Analyzing your resume..."):
                                    ai_feedback = analyze_resume_with_ai(resume_text)
                                    st.markdown("### 🤖 AI Feedback")
                                    st.info(ai_feedback)
                            
                            # Save analysis with default score
                            if st.button("💾 Save Analysis"):
                                if save_resume_analysis(
                                    st.session_state.user_email,
                                    resume_text,
                                    75,  # Default score
                                    best_positions[0] if best_positions else {}
                                ):
                                    st.success("✅ Analysis saved successfully!")
                
                with col2:
                    st.markdown("### 💡 Quick Tips")
                    st.markdown("""
                    <div class="feature-box">
                        <p><strong>✅ Do's:</strong></p>
                        <ul>
                            <li>Use clear section headers</li>
                            <li>Include relevant keywords</li>
                            <li>Quantify achievements</li>
                            <li>Keep format simple</li>
                        </ul>
                        
                        <p><strong>❌ Don'ts:</strong></p>
                        <ul>
                            <li>Avoid images/graphics</li>
                            <li>No fancy fonts</li>
                            <li>Don't use tables</li>
                            <li>Avoid headers/footers</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # TAB 2: JD Matching
            with tab2:
                st.markdown("### 🎯 Job Description Matching")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📄 Upload Resume**")
                    resume_file = st.file_uploader("Resume", type=['pdf', 'docx'], key="jd_resume")
                
                with col2:
                    st.markdown("**📋 Upload Job Description**")
                    jd_file = st.file_uploader("Job Description", type=['pdf', 'docx', 'txt'], key="jd_file")
                
                if resume_file and jd_file:
                    if st.button("🔍 Calculate ATS Score", type="primary"):
                        with st.spinner("Analyzing match..."):
                            # Extract texts
                            if resume_file.type == "application/pdf":
                                resume_text = extract_text_from_pdf(resume_file)
                            else:
                                resume_text = extract_text_from_docx(resume_file)
                            
                            if jd_file.type == "application/pdf":
                                jd_text = extract_text_from_pdf(jd_file)
                            elif jd_file.type == "text/plain":
                                jd_text = jd_file.read().decode("utf-8")
                            else:
                                jd_text = extract_text_from_docx(jd_file)
                            
                            # Calculate ATS score
                            ats_score = calculate_ats_score(resume_text, jd_text)
                            
                            # Display score
                            score_color = "#4CAF50" if ats_score >= 80 else "#FF9800" if ats_score >= 60 else "#F44336"
                            
                            st.markdown(f"""
                            <div class="resume-card">
                                <h2 style="text-align: center;">ATS Compatibility Score</h2>
                                <div style="background: #f0f0f0; border-radius: 10px; padding: 5px; margin: 20px 0;">
                                    <div style="background: {score_color}; width: {ats_score}%; padding: 15px; 
                                                border-radius: 8px; color: white; text-align: center; font-size: 1.5rem; font-weight: bold;">
                                        {ats_score}%
                                    </div>
                                </div>
                                <p style="text-align: center; font-size: 1.1rem;">
                                    {'🎉 Excellent Match!' if ats_score >= 80 else '👍 Good Match!' if ats_score >= 60 else '⚠️ Needs Improvement'}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Get suggestions
                            suggestions = get_resume_suggestions(resume_text, jd_text, ats_score)
                            
                            st.markdown("### 💡 Improvement Suggestions")
                            st.info(suggestions)
                            
                            # Detect positions
                            best_positions = detect_best_position(resume_text)
                            
                            # Save to database
                            if save_resume_analysis(
                                st.session_state.user_email,
                                resume_text,
                                ats_score,
                                best_positions[0] if best_positions else {}
                            ):
                                st.success("✅ Analysis saved to your profile!")
            
            # TAB 3: Masterclass
            with tab3:
                st.markdown("### 🎓 Resume Writing Masterclass")
                
                st.markdown("""
                <div class="info-card">
                    <h3>📚 Learn Resume Writing Best Practices</h3>
                    <p>Master the art of creating ATS-friendly resumes that get you noticed!</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Video tutorials section
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="resume-card">
                        <h4>🎬 Module 1: ATS Basics</h4>
                        <p>Understanding how ATS systems work and what they look for</p>
                        <ul>
                            <li>What is ATS?</li>
                            <li>How companies use ATS</li>
                            <li>Common ATS mistakes</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="resume-card">
                        <h4>📝 Module 3: Skills Section</h4>
                        <p>How to effectively showcase your skills</p>
                        <ul>
                            <li>Technical skills</li>
                            <li>Soft skills</li>
                            <li>Skill organization</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="resume-card">
                        <h4>✏️ Module 2: Formatting Tips</h4>
                        <p>Create a clean, ATS-friendly format</p>
                        <ul>
                            <li>Font selection</li>
                            <li>Section organization</li>
                            <li>File format best practices</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="resume-card">
                        <h4>🏆 Module 4: Keywords & Optimization</h4>
                        <p>Optimize your resume for specific jobs</p>
                        <ul>
                            <li>Finding job keywords</li>
                            <li>Natural keyword integration</li>
                            <li>Action verbs and quantification</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.info("🎥 Video tutorials and detailed guides coming soon!")
            
            # TAB 4: About
            with tab4:
                st.markdown("### ℹ️ About AI Resume Platform")
                
                st.markdown("""
                <div class="info-card">
                    <h3>🎯 Our Mission</h3>
                    <p>We help job seekers create ATS-friendly resumes that get noticed by recruiters and land interviews.</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class="stats-box">
                        <h2>10,000+</h2>
                        <p>Resumes Analyzed</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="stats-box">
                        <h2>85%</h2>
                        <p>Success Rate</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class="stats-box">
                        <h2>500+</h2>
                        <p>Partner Companies</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-card">
                    <h3>✨ Features</h3>
                    <ul>
                        <li>🤖 AI-Powered Resume Analysis</li>
                        <li>📊 Real-time ATS Score Calculation</li>
                        <li>🎯 Job Position Recommendations</li>
                        <li>💡 Personalized Improvement Suggestions</li>
                        <li>📈 Resume Performance Tracking</li>
                        <li>🎓 Educational Masterclass Content</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-card">
                    <h3>📧 Contact Us</h3>
                    <p><strong>Email:</strong> support@airesume.com</p>
                    <p><strong>Phone:</strong> +91 1234567890</p>
                    <p><strong>Address:</strong> Mumbai, India</p>
                </div>
                """, unsafe_allow_html=True)
            
            # TAB 5: Profile
            with tab5:
                st.markdown("### 👤 Your Profile")
                
                if user:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("""
                        <div class="resume-card">
                            <h3>👤 Personal Info</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="info-card">
                            <p><strong>Name:</strong><br>{user.get('name', 'N/A')}</p>
                            <p><strong>Email:</strong><br>{user.get('email', 'N/A')}</p>
                            <p><strong>Phone:</strong><br>{user.get('phone', 'N/A')}</p>
                            <p><strong>College:</strong><br>{user.get('college', 'N/A')}</p>
                            <p><strong>Registered:</strong><br>{user.get('registration_date', 'N/A').strftime('%d %b %Y') if user.get('registration_date') else 'N/A'}</p>
                            <p><strong>Payment Status:</strong><br>{'✅ Active' if user.get('payment_status') == 'completed' else '⏳ Pending'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### 📊 Your Analytics")
                        
                        # Get user's saved analyses
                        user_analyses = list(resume_collection.find({"user_email": st.session_state.user_email}))
                        
                        if user_analyses:
                            latest = user_analyses[0]
                            
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.metric("Latest ATS Score", f"{latest.get('ats_score', 0)}%")
                            
                            with col_b:
                                st.metric("Best Position", latest.get('best_position', {}).get('position', 'N/A'))
                            
                            with col_c:
                                st.metric("Match Confidence", latest.get('best_position', {}).get('confidence', 'N/A'))
                            
                            st.markdown("### 📈 Analysis History")
                            
                            for idx, analysis in enumerate(user_analyses[:5]):  # Show last 5
                                st.markdown(f"""
                                <div class="resume-card">
                                    <h4>Analysis #{idx+1}</h4>
                                    <p><strong>Date:</strong> {analysis.get('analysis_date', 'N/A').strftime('%d %b %Y, %I:%M %p') if analysis.get('analysis_date') else 'N/A'}</p>
                                    <p><strong>ATS Score:</strong> {analysis.get('ats_score', 'N/A')}%</p>
                                    <p><strong>Position:</strong> {analysis.get('best_position', {}).get('position', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No analysis history yet. Upload a resume to get started!")
    
    # ================== HR LOGIN PAGE ==================
    elif st.session_state.page == 'hr_login':
        # Back button
        if st.button("← Back to Home"):
            st.session_state.page = 'home'
            st.rerun()
        
        st.markdown("""
        <div class="main-header hr-section">
            <h1>💼 HR Portal</h1>
            <p>Access Talent Database & Recruitment Tools</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Centered login form
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            tab1, tab2 = st.tabs(["🔐 Login", "📝 Register Company"])
            
            with tab1:
                st.markdown("<br>", unsafe_allow_html=True)
                hr_email = st.text_input("HR Email Address", key="hr_login_email")
                company_name = st.text_input("Company Name", key="hr_login_company")
                
                if st.button("Login to HR Portal", type="primary"):
                    if hr_email and company_name:
                        success, hr_user = verify_hr_login(hr_email, company_name)
                        if success:
                            st.session_state.hr_logged_in = True
                            st.session_state.hr_email = hr_email
                            st.session_state.hr_company = company_name
                            st.session_state.page = 'hr_dashboard'
                            st.success("Login successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Invalid credentials! Please check your details or register.")
                    else:
                        st.warning("Please fill all fields")
            
            with tab2:
                st.markdown("<br>", unsafe_allow_html=True)
                company_name = st.text_input("Company Name", key="hr_reg_company")
                hr_name = st.text_input("HR Manager Name", key="hr_reg_name")
                hr_email = st.text_input("Official Email", key="hr_reg_email")
                phone = st.text_input("Contact Number", key="hr_reg_phone")
                
                company_size = st.selectbox(
                    "Company Size",
                    ["1-10", "11-50", "51-200", "201-500", "500+"]
                )
                
                industry = st.selectbox(
                    "Industry",
                    ["IT/Software", "Finance", "Healthcare", "Education", "Manufacturing", "Retail", "Other"]
                )
                
                if st.button("Register Company", type="primary"):
                    if company_name and hr_name and hr_email and phone:
                        success, message = save_hr_registration(
                            company_name, hr_name, hr_email, phone, company_size, industry
                        )
                        if success:
                            st.success(message + " Please login to continue.")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("Please fill all required fields")
    
    # ================== HR DASHBOARD ==================
    elif st.session_state.page == 'hr_dashboard':
        if not st.session_state.hr_logged_in:
            st.session_state.page = 'hr_login'
            st.rerun()
        
        # Logout button
        col1, col2, col3 = st.columns([6, 1, 1])
        with col3:
            if st.button("🚪 Logout"):
                st.session_state.hr_logged_in = False
                st.session_state.hr_email = None
                st.session_state.page = 'home'
                st.rerun()
        
        # HR Dashboard Header - CENTERED
        st.markdown(f"""
        <div class="centered-container">
            <div class="main-header hr-section">
                <h1>💼 HR Dashboard</h1>
                <p>{st.session_state.hr_company} - Talent Search Portal</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Search Filters - CENTERED
        st.markdown('<div class="centered-container">', unsafe_allow_html=True)
        st.markdown("## 🔍 Search Candidates")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            position_filter = st.selectbox(
                "🎯 Position",
                ["All Positions", "Data Analyst", "Data Scientist", "Software Engineer", 
                 "Project Manager", "Business Analyst", "DevOps Engineer", "Frontend Developer",
                 "Backend Developer", "Marketing Manager", "HR Manager", "Sales Executive",
                 "UI/UX Designer", "Product Manager", "QA Engineer", "Content Writer"]
            )
        
        with col2:
            ats_filter = st.selectbox(
                "📊 ATS Score Range",
                ["All Scores", "90-100 (Excellent)", "80-89 (Very Good)", "70-79 (Good)", "60-69 (Fair)", "Below 60"]
            )
            
            # Convert filter to minimum score
            if ats_filter == "90-100 (Excellent)":
                min_ats = 90
            elif ats_filter == "80-89 (Very Good)":
                min_ats = 80
            elif ats_filter == "70-79 (Good)":
                min_ats = 70
            elif ats_filter == "60-69 (Fair)":
                min_ats = 60
            elif ats_filter == "Below 60":
                min_ats = 0
            else:
                min_ats = 0
        
        with col3:
            sort_by = st.selectbox(
                "🔄 Sort By",
                ["ATS Score (High to Low)", "ATS Score (Low to High)", "Recent First"]
            )
        
        # Search Button
        if st.button("🔍 Search Candidates", type="primary"):
            with st.spinner("Searching candidates..."):
                time.sleep(1)
                st.success("Search completed!")
        
        # Get total count first
        total_resumes = resume_collection.count_documents({})
        st.info(f"📊 Total resumes in database: {total_resumes}")
        
        # Get filtered resumes
        filtered_resumes = get_filtered_resumes(
            position=position_filter,
            min_ats_score=min_ats
        )
        
        # Display Results - CENTERED
        st.markdown(f"## 📋 Search Results ({len(filtered_resumes)} candidates found)")
        
        if len(filtered_resumes) == 0:
            st.warning("⚠️ No candidates found matching your criteria.")
            st.info("""
            💡 **Tips:**
            - Try selecting 'All Positions' to see all candidates
            - Lower the ATS score filter
            - Make sure students have uploaded their resumes
            """)
        else:
            # Tabs for different views
            tab1, tab2 = st.tabs(["📄 All Candidates", "⭐ Shortlisted"])
            
            with tab1:
                for idx, resume in enumerate(filtered_resumes):
                    with st.expander(f"🎯 Candidate #{idx+1} - {resume.get('best_position', {}).get('position', 'N/A')} | ATS: {resume.get('ats_score', 'N/A')}%", expanded=False):
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("### 📊 Candidate Overview")
                            
                            # Position Badge
                            if 'best_position' in resume:
                                pos_data = resume['best_position']
                                confidence_color = "badge-green" if pos_data.get('confidence') == 'High' else "badge-orange" if pos_data.get('confidence') == 'Medium' else "badge-blue"
                                
                                st.markdown(f"""
                                <div style="margin: 15px 0;">
                                    <span class="position-badge {confidence_color}">
                                        {pos_data.get('position', 'N/A')}
                                    </span>
                                    <span class="position-badge badge-blue">
                                        {pos_data.get('match_percentage', 0)}% Match
                                    </span>
                                    <span class="position-badge badge-orange">
                                        {pos_data.get('keyword_count', 0)} Skills
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # ATS Score
                            ats_score = resume.get('ats_score', 0)
                            score_color = "#4CAF50" if ats_score >= 80 else "#FF9800" if ats_score >= 60 else "#F44336"
                            
                            st.markdown(f"""
                            <div style="margin: 20px 0;">
                                <h4>ATS Compatibility Score</h4>
                                <div style="background: #f0f0f0; border-radius: 10px; padding: 3px;">
                                    <div style="background: {score_color}; width: {ats_score}%; padding: 10px; 
                                                border-radius: 8px; color: white; text-align: center; font-weight: bold;">
                                        {ats_score}%
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Resume Preview
                            st.markdown("**📄 Resume Preview:**")
                            resume_text = resume.get('resume_text', '')
                            preview_text = resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
                            st.text_area("", preview_text, height=150, key=f"preview_{idx}", disabled=True)
                        
                        with col2:
                            st.markdown("### 📞 Contact Information")
                            
                            # Get student info
                            student_email = resume.get('user_email', 'Not available')
                            
                            # Try to get student details from user collection
                            try:
                                student_info = collection.find_one({"email": student_email})
                                if student_info:
                                    st.markdown(f"""
                                    <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                        <p><strong>👤 Name:</strong><br>{student_info.get('name', 'N/A')}</p>
                                        <p><strong>📧 Email:</strong><br>{student_info.get('email', 'N/A')}</p>
                                        <p><strong>📱 Phone:</strong><br>{student_info.get('phone', 'N/A')}</p>
                                        <p><strong>🎓 College:</strong><br>{student_info.get('college', 'N/A')}</p>
                                        <p><strong>📅 Registered:</strong><br>{student_info.get('registration_date', 'N/A').strftime('%d %b %Y') if student_info.get('registration_date') else 'N/A'}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.info("Contact details available after shortlisting")
                            except:
                                st.info("Contact details available after shortlisting")
                            
                            st.markdown("### 🎯 Actions")
                            
                            # Shortlist button
                            resume_id = str(resume.get('_id', ''))
                            if st.button(f"⭐ Shortlist", key=f"shortlist_{idx}"):
                                if shortlist_candidate(st.session_state.hr_email, student_email, resume_id):
                                    st.success("✅ Candidate shortlisted!")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.warning("Already shortlisted!")
                            
                            # Download button (placeholder)
                            if st.button(f"📥 Download Resume", key=f"download_{idx}"):
                                st.success("📄 Resume download started!")
                            
                            # Contact button
                            if st.button(f"📧 Send Email", key=f"contact_{idx}"):
                                st.info(f"Opening email client to: {student_email}")
            
            with tab2:
                shortlisted = get_shortlisted_candidates(st.session_state.hr_email)
                
                if len(shortlisted) == 0:
                    st.info("No candidates shortlisted yet. Start shortlisting from 'All Candidates' tab!")
                else:
                    st.success(f"✅ You have shortlisted {len(shortlisted)} candidates")
                    
                    for idx, item in enumerate(shortlisted):
                        student_email = item.get('student_email')
                        
                        # Get resume details
                        try:
                            resume = resume_collection.find_one({"user_email": student_email})
                            if resume:
                                st.markdown(f"""
                                <div class="resume-card">
                                    <h4>🎯 Shortlisted Candidate #{idx+1}</h4>
                                    <p><strong>Email:</strong> {student_email}</p>
                                    <p><strong>Position:</strong> {resume.get('best_position', {}).get('position', 'N/A')}</p>
                                    <p><strong>ATS Score:</strong> {resume.get('ats_score', 'N/A')}%</p>
                                    <p><strong>Shortlisted on:</strong> {item.get('shortlisted_date', 'N/A').strftime('%d %b %Y, %I:%M %p') if item.get('shortlisted_date') else 'N/A'}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        except:
                            pass
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>Need help? Contact support@resumeplatform.com</p>
            <p style="font-size: 0.9rem;">© 2024 AI Resume Platform - HR Portal</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
