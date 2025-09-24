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
import google.generativeai as genai
import io
import qrcode
from PIL import Image
from pymongo import MongoClient
import uuid
from datetime import datetime, timedelta

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

# ================== CONFIG ==================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="AI Resume + Job Prep Tool", layout="wide")
st.title("🚀 AI-Powered Resume + Job Prep Platform")

# ================== SESSION STATE ==================
if "resume_uploads" not in st.session_state:
    st.session_state.resume_uploads = 0
if "subscribed" not in st.session_state:
    st.session_state.subscribed = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "payment_order_id" not in st.session_state:
    st.session_state.payment_order_id = None

# ================== DATABASE HELPER FUNCTIONS ==================
def save_user_basic_info(name, email, phone):
    """User ki basic info save karta hai"""
    try:
        user_data = {
            "name": name, 
            "email": email, 
            "phone": phone,
            "registration_date": datetime.now(),
            "subscription_status": "free",
            "subscription_expiry": None
        }
        
        # Check if user already exists
        existing_user = collection.find_one({"email": email})
        if existing_user:
            collection.update_one(
                {"email": email},
                {"$set": user_data}
            )
        else:
            collection.insert_one(user_data)
        
        st.session_state.user_email = email
        st.session_state.user_name = name
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False

def create_payment_order(user_email, user_name, phone, amount, plan_type):
    """Payment order create karta hai aur unique order ID generate karta hai"""
    try:
        order_id = f"ORD_{uuid.uuid4().hex[:8].upper()}"
        
        order_data = {
            "order_id": order_id,
            "user_email": user_email,
            "user_name": user_name,
            "phone": phone,
            "amount": amount,
            "plan_type": plan_type,
            "status": "pending",
            "created_date": datetime.now(),
            "expires_at": datetime.now() + timedelta(minutes=30)  # 30 min expiry
        }
        
        pending_payments.insert_one(order_data)
        return order_id
    except Exception as e:
        st.error(f"Order Creation Error: {e}")
        return None

def verify_payment_status(order_id):
    """Payment status check karta hai (Demo ke liye manual verification)"""
    try:
        order = pending_payments.find_one({"order_id": order_id})
        if order:
            return order.get("status", "pending")
        return "not_found"
    except Exception:
        return "error"

def complete_payment(order_id):
    """Payment complete karta hai aur subscription activate karta hai"""
    try:
        order = pending_payments.find_one({"order_id": order_id})
        if not order:
            return False
            
        # Payment record save karo
        payment_data = {
            "order_id": order_id,
            "user_email": order["user_email"],
            "user_name": order["user_name"],
            "phone": order["phone"],
            "amount": order["amount"],
            "plan_type": order["plan_type"],
            "payment_method": "UPI/QR",
            "payment_date": datetime.now(),
            "status": "completed"
        }
        
        payment_collection.insert_one(payment_data)
        
        # Subscription activate karo
        expiry_date = datetime.now() + timedelta(days=365 if order["amount"] >= 999 else 30)
        
        collection.update_one(
            {"email": order["user_email"]},
            {"$set": {
                "subscription_status": "premium",
                "subscription_expiry": expiry_date,
                "last_payment": datetime.now()
            }}
        )
        
        # Pending order ko complete mark karo
        pending_payments.update_one(
            {"order_id": order_id},
            {"$set": {"status": "completed"}}
        )
        
        return True
    except Exception as e:
        st.error(f"Payment Completion Error: {e}")
        return False

def save_payment_info(user_email, user_name, phone, amount, payment_method="QR"):
    """Payment information save karta hai"""
    try:
        payment_data = {
            "user_email": user_email,
            "user_name": user_name,
            "phone": phone,
            "amount": amount,
            "payment_method": payment_method,
            "payment_date": datetime.now(),
            "subscription_type": "premium" if amount >= 199 else "basic",
            "status": "completed"
        }
        
        payment_collection.insert_one(payment_data)
        
        # Update user subscription status
        expiry_date = datetime.now() + timedelta(days=365 if amount >= 999 else 30)
        
        collection.update_one(
            {"email": user_email},
            {"$set": {
                "subscription_status": "premium",
                "subscription_expiry": expiry_date,
                "last_payment": datetime.now()
            }}
        )
        
        st.session_state.subscribed = True
        return True
    except Exception as e:
        st.error(f"Payment Save Error: {e}")
        return False

def check_user_subscription(user_email):
    """User ka subscription status check karta hai"""
    try:
        user = collection.find_one({"email": user_email})
        if user and user.get("subscription_status") == "premium":
            expiry = user.get("subscription_expiry")
            if expiry and datetime.now() < expiry:
                st.session_state.subscribed = True
                return True
        return False
    except Exception:
        return False

def get_user_profile(user_email):
    """User ka complete profile return karta hai"""
    try:
        user = collection.find_one({"email": user_email})
        if user:
            # Count total analyses
            resume_count = resume_collection.count_documents({"user_email": user_email})
            jd_count = jd_collection.count_documents({"user_email": user_email})
            payment_history = list(payment_collection.find({"user_email": user_email}).sort("payment_date", -1))
            
            return {
                "user_info": user,
                "resume_analyses": resume_count,
                "jd_matches": jd_count,
                "payments": payment_history
            }
        return None
    except Exception as e:
        st.error(f"Profile Fetch Error: {e}")
        return None

def save_resume_analysis(user_email, resume_text, ats_score, feedback, filename=None):
    """Resume analysis data save karta hai"""
    try:
        analysis_data = {
            "user_email": user_email,
            "resume_text": resume_text[:1000],  # first 1000 chars save karenge space bachane ke liye
            "ats_score": ats_score,
            "ai_feedback": feedback,
            "filename": filename,
            "analysis_date": datetime.now(),
            "analysis_type": "resume_analyzer"
        }
        
        resume_collection.insert_one(analysis_data)
        return True
    except Exception as e:
        st.error(f"Resume Analysis Save Error: {e}")
        return False

def save_jd_matching_data(user_email, similarity_score, resume_skills, jd_skills, missing_skills, ai_suggestions):
    """JD matching data save karta hai"""
    try:
        jd_data = {
            "user_email": user_email,
            "similarity_score": similarity_score,
            "resume_skills": resume_skills,
            "required_skills": jd_skills,
            "missing_skills": missing_skills,
            "ai_suggestions": ai_suggestions,
            "matching_date": datetime.now(),
            "analysis_type": "jd_matcher"
        }
        
        jd_collection.insert_one(jd_data)
        return True
    except Exception as e:
        st.error(f"JD Matching Save Error: {e}")
        return False

def get_user_history(user_email):
    """User ka analysis history return karta hai"""
    try:
        resume_history = list(resume_collection.find({"user_email": user_email}).sort("analysis_date", -1).limit(5))
        jd_history = list(jd_collection.find({"user_email": user_email}).sort("matching_date", -1).limit(5))
        return resume_history, jd_history
    except Exception as e:
        st.error(f"History Fetch Error: {e}")
        return [], []

# ================== ORIGINAL HELPER FUNCTIONS ==================
def extract_text_from_pdf(file_bytes_io) -> str:
    try:
        file_bytes_io.seek(0)
        with pdfplumber.open(file_bytes_io) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages).strip()
    except Exception:
        try:
            file_bytes_io.seek(0)
            reader = PdfReader(file_bytes_io)
            return "\n".join([p.extract_text() or "" for p in reader.pages]).strip()
        except Exception:
            return ""

def extract_text_from_docx(file_bytes_io) -> str:
    try:
        file_bytes_io.seek(0)
        doc = Document(file_bytes_io)
        return "\n".join([p.text for p in doc.paragraphs if p.text]).strip()
    except Exception:
        return ""

def extract_text_from_uploaded_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    ext = name.split(".")[-1]
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    if ext == "pdf":
        return extract_text_from_pdf(uploaded_file) or ""
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(uploaded_file) or ""
    else:
        try:
            uploaded_file.seek(0)
            return uploaded_file.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""

def match_resume_jd_tfidf(resume_text: str, jd_text: str, top_k: int = 15):
    if not jd_text or not resume_text:
        return 0.0, [], [], []
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X = vect.fit_transform([jd_text, resume_text])
    sim = float(cosine_similarity(X[0], X[1])[0][0])
    sim_pct = round(sim * 100, 1)

    feature_names = vect.get_feature_names_out()
    jd_vec = X[0].toarray().flatten()
    resume_vec = X[1].toarray().flatten()

    top_indices = jd_vec.argsort()[::-1][:top_k]
    top_terms = [feature_names[i] for i in top_indices if jd_vec[i] > 0]

    present = [term for term in top_terms if resume_vec[feature_names.tolist().index(term)] > 0]
    missing = [term for term in top_terms if term not in present]

    return sim_pct, top_terms, present, missing

def gemini_insights(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"(Gemini API Error: {e})"

# ================== MENU ==================
menu = ["🏠 Home", "📂 Resume Analyzer", "📄 JD Matcher", "🎓 Masterclass", "💳 Subscription", "👤 Profile", "ℹ About Us"]
choice = st.sidebar.selectbox("Navigate", menu)

# Enhanced User Profile Sidebar
if st.session_state.user_email and st.session_state.user_name:
    with st.sidebar:
        st.markdown("---")
        
        # Profile Header with nice styling
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 15px; text-align: center;">
                <div style="color: white; font-size: 18px; font-weight: bold; margin-bottom: 5px;">
                    👋 Welcome Back!
                </div>
                <div style="color: #e8f4fd; font-size: 14px; margin-bottom: 8px;">
                    {st.session_state.user_name}
                </div>
                <div style="color: #d1ecf1; font-size: 12px;">
                    {st.session_state.user_email}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Subscription Status
        subscription_status = "🔓 Premium" if st.session_state.subscribed else "🔒 Free"
        status_color = "#4CAF50" if st.session_state.subscribed else "#FF9800"
        
        st.markdown(
            f"""
            <div style="background-color: {status_color}20; border-left: 4px solid {status_color}; 
                        padding: 10px; margin-bottom: 15px; border-radius: 5px;">
                <div style="color: {status_color}; font-weight: bold; font-size: 14px;">
                    {subscription_status}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Quick Stats
        if st.button("📊 View Full Profile"):
            st.session_state.show_profile = True
        
        # Recent Activity (simplified)
        try:
            resume_hist, jd_hist = get_user_history(st.session_state.user_email)
            
            st.markdown("📈 Recent Activity**")
            if resume_hist:
                latest = resume_hist[0]
                st.markdown(f"🔸 Last Resume: {latest.get('ats_score', 'N/A')}%")
            
            if jd_hist:
                latest = jd_hist[0]
                st.markdown(f"🔸 Last JD Match: {latest.get('similarity_score', 'N/A')}%")
                
        except:
            pass

# ================== HOME ==================
# ================== HOME (Enhanced) ==================
if choice == "🏠 Home":
    # Custom CSS for enhanced home page
    st.markdown("""
    <style>
    .main-hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 50px 30px;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .main-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 15px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
        margin-bottom: 25px;
        position: relative;
        z-index: 1;
    }
    
    .hero-cta {
        position: relative;
        z-index: 1;
    }
    
    .feature-card {
        background: white;
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-top: 4px solid;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.15);
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .feature-card:hover::before {
        left: 100%;
    }
    
    .resume-card { border-top-color: #4CAF50; }
    .jd-card { border-top-color: #2196F3; }
    .masterclass-card { border-top-color: #FF9800; }
    .subscription-card { border-top-color: #9C27B0; }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 15px;
        display: block;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
    
    .feature-description {
        color: #666;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    
    .stats-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 20px;
        padding: 40px 20px;
        margin: 40px 0;
        text-align: center;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 30px;
        margin-top: 30px;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        display: block;
        margin-bottom: 5px;
    }
    
    .stat-label {
        color: #666;
        font-size: 1rem;
        font-weight: 500;
    }
    
    .user-form-container {
        background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 40px;
        border: 2px solid #FFB74D;
        position: relative;
    }
    
    .form-title {
        color: #E65100;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 15px;
        text-align: center;
    }
    
    .form-subtitle {
        color: #666;
        text-align: center;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }
    
    .testimonial-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border-left: 4px solid #4CAF50;
    }
    
    .testimonial-text {
        font-style: italic;
        color: #555;
        margin-bottom: 15px;
        line-height: 1.6;
    }
    
    .testimonial-author {
        font-weight: bold;
        color: #4CAF50;
    }
    
    .quick-actions {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 30px 0;
    }
    
    .action-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    
    .action-item {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .action-item:hover {
        transform: scale(1.05);
    }
    
    .news-ticker {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 30px;
        overflow: hidden;
        position: relative;
    }
    
    .ticker-content {
        animation: scroll-left 20s linear infinite;
        white-space: nowrap;
    }
    
    @keyframes scroll-left {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # News Ticker
    st.markdown("""
    <div class="news-ticker">
        <div class="ticker-content">
            🎉 New Feature Alert: AI-powered interview prep now available! | 
            📈 500+ users got hired this month using our platform | 
            🎓 Join our free webinar on "ATS Resume Secrets" this Friday |
            💡 Premium users get 90% better job match rates
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Hero Section
    st.markdown("""
    <div class="main-hero">
        <div class="hero-title">🚀 AI-Powered Career Platform</div>
        <div class="hero-subtitle">Transform Your Career with Smart Resume Analysis, Job Matching & Expert Guidance</div>
        <div class="hero-cta">
            <p>Join 10,000+ professionals who landed their dream jobs</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ----------- USER FORM (Enhanced) -----------
    if not st.session_state.user_email:
        st.markdown("""
        <div class="user-form-container">
            <div class="form-title">🌟 Start Your Career Journey</div>
            <div class="form-subtitle">Join thousands of successful professionals. Get personalized insights in 30 seconds!</div>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced form with better layout
        with st.form("user_details_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("👤 Full Name", placeholder="Enter your full name")
                email = st.text_input("📧 Email Address", placeholder="your.email@company.com")
            with col2:
                phone = st.text_input("📱 Phone Number", placeholder="+91 XXXXX XXXXX")
                experience = st.selectbox("💼 Experience Level", 
                    ["Select Experience", "0-1 years (Fresher)", "1-3 years", "3-5 years", "5-10 years", "10+ years"])
            
            industry = st.selectbox("🏢 Industry/Domain", 
                ["Select Industry", "Information Technology", "Data Science/Analytics", "Marketing", 
                 "Finance", "Healthcare", "Education", "Engineering", "Consulting", "Other"])
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submitted = st.form_submit_button("✨ Join Now & Get Free Analysis", use_container_width=True)

        if submitted:
            if not name or not email or not phone or experience == "Select Experience":
                st.error("⚠️ Please fill in all required details to continue.")
            else:
                if save_user_basic_info(name, email, phone):
                    st.success(f"🎉 Welcome aboard, {name}! Your career transformation starts now.")
                    st.balloons()
                    # Check subscription status
                    check_user_subscription(email)
                    time.sleep(1)
                    st.rerun()
    else:
        # Welcome back section for existing users
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                   border-radius: 20px; padding: 30px; text-align: center; margin-bottom: 30px;
                   border: 2px solid #4CAF50;">
            <h2 style="color: #2e7d32; margin: 0;">👋 Welcome Back, {st.session_state.user_name}!</h2>
            <p style="color: #388e3c; font-size: 1.1rem; margin-top: 10px;">
                Ready to take the next step in your career journey?
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check subscription status
        check_user_subscription(st.session_state.user_email)

    # Platform Statistics
    st.markdown("""
    <div class="stats-section">
        <h2 style="color: #333; margin-bottom: 10px;">📊 Platform Success Stories</h2>
        <p style="color: #666;">Real impact, real results from our community</p>
        <div class="stats-grid">
            <div class="stat-item">
                <span class="stat-number">10,000+</span>
                <span class="stat-label">Users Registered</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">25,000+</span>
                <span class="stat-label">Resumes Analyzed</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">95%</span>
                <span class="stat-label">Success Rate</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">500+</span>
                <span class="stat-label">Jobs This Month</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced Features Section with Interactive Cards
    st.markdown("## 🎯 Explore Our Platform")
    
    # Features in 2x2 grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card resume-card">
            <span class="feature-icon">📊</span>
            <div class="feature-title">Smart Resume Analyzer</div>
            <div class="feature-description">
                Get instant ATS compatibility scores, keyword optimization, and AI-powered suggestions 
                to make your resume stand out to recruiters.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Try Resume Analyzer", key="try_resume", use_container_width=True):
            st.session_state.redirect_page = "📂 Resume Analyzer"
            st.success("Redirecting to Resume Analyzer...")
            time.sleep(1)
            st.rerun()
        
        st.markdown("""
        <div class="feature-card masterclass-card">
            <span class="feature-icon">🎓</span>
            <div class="feature-title">Expert Masterclasses</div>
            <div class="feature-description">
                Learn from industry leaders at Google, Microsoft, Amazon. Get insider tips on 
                interviews, career growth, and skill development.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📚 Browse Masterclasses", key="try_masterclass", use_container_width=True):
            st.session_state.redirect_page = "🎓 Masterclass"
            st.success("Opening Masterclasses...")
            time.sleep(1)
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="feature-card jd-card">
            <span class="feature-icon">🎯</span>
            <div class="feature-title">Job Description Matcher</div>
            <div class="feature-description">
                Compare your resume with job descriptions. Get similarity scores, missing skills analysis, 
                and tailored improvement suggestions.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Try JD Matcher", key="try_jd", use_container_width=True):
            st.session_state.redirect_page = "📄 JD Matcher"
            st.success("Opening JD Matcher...")
            time.sleep(1)
            st.rerun()
        
        st.markdown("""
        <div class="feature-card subscription-card">
            <span class="feature-icon">💎</span>
            <div class="feature-title">Premium Features</div>
            <div class="feature-description">
                Unlimited analyses, priority support, exclusive masterclasses, 1-on-1 career coaching, 
                and advanced AI insights.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✨ Explore Premium", key="try_premium", use_container_width=True):
            st.session_state.redirect_page = "💎 Subscription"
            st.success("Opening Premium Plans...")
            time.sleep(1)
            st.rerun()

    # Quick Actions Section
    st.markdown("""
    <div class="quick-actions">
        <h3 style="color: #1976d2; text-align: center; margin-bottom: 10px;">⚡ Quick Actions</h3>
        <p style="text-align: center; color: #666; margin-bottom: 0;">Get started with these popular features</p>
    </div>
    """, unsafe_allow_html=True)
    
    quick_cols = st.columns(4)
    
    with quick_cols[0]:
        if st.button("📈 Career Assessment", use_container_width=True):
            st.info("🔍 **Free Career Assessment**\n\n1. Skills evaluation\n2. Industry fit analysis\n3. Growth recommendations\n4. Salary insights")
    
    with quick_cols[1]:
        if st.button("🎯 Interview Prep", use_container_width=True):
            st.info("💡 **Interview Preparation**\n\n• Common questions bank\n• Behavioral interview tips\n• Technical skill assessment\n• Mock interview sessions")
    
    with quick_cols[2]:
        if st.button("📊 Salary Insights", use_container_width=True):
            st.info("💰 **Industry Salary Data**\n\n• Role-based salary ranges\n• Location adjustments\n• Experience level impact\n• Negotiation strategies")
    
    with quick_cols[3]:
        if st.button("🌐 Job Market Trends", use_container_width=True):
            st.info("📈 **Current Market Trends**\n\n• In-demand skills\n• Growing industries\n• Remote work insights\n• Future job predictions")

    # Success Stories / Testimonials
    st.markdown("## 🌟 Success Stories")
    
    testimonials = [
        {
            "text": "This platform helped me increase my interview calls by 300%! The ATS optimization was a game-changer.",
            "author": "Priya S., Software Engineer at Google"
        },
        {
            "text": "The masterclasses gave me insights I couldn't find anywhere else. Landed my dream job in data science!",
            "author": "Rahul M., Data Scientist at Microsoft"
        },
        {
            "text": "Premium subscription paid for itself with the first job offer. The 1-on-1 coaching was incredible.",
            "author": "Anita K., Product Manager at Amazon"
        }
    ]
    
    test_cols = st.columns(3)
    for idx, testimonial in enumerate(testimonials):
        with test_cols[idx]:
            st.markdown(f"""
            <div class="testimonial-card">
                <div class="testimonial-text">"{testimonial['text']}"</div>
                <div class="testimonial-author">— {testimonial['author']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Call-to-Action Section
    if st.session_state.user_email and not st.session_state.subscribed:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 40px; border-radius: 20px; text-align: center; margin-top: 40px;">
            <h3 style="margin-top: 0;">🚀 Ready to Accelerate Your Career?</h3>
            <p style="font-size: 1.1rem; opacity: 0.9; margin-bottom: 25px;">
                Join thousands of professionals who upgraded to Premium and achieved their career goals 3x faster.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("💎 Upgrade to Premium Now", use_container_width=True, key="cta_premium"):
            st.session_state.redirect_page = "💎 Subscription"
            st.success("Redirecting to Premium plans...")
            time.sleep(1)
            st.rerun()

    # Footer with additional resources
    st.markdown("---")
    st.markdown("## 📚 Additional Resources")
    
    resource_cols = st.columns(3)
    
    with resource_cols[0]:
        st.markdown("""
        **📖 Learning Resources**
        - [Resume Writing Guide](https://example.com)
        - [Interview Preparation](https://example.com)
        - [Career Change Roadmap](https://example.com)
        - [Salary Negotiation Tips](https://example.com)
        """)
    
    with resource_cols[1]:
        st.markdown("""
        **🔗 Useful Links**
        - [LinkedIn Profile Optimization](https://example.com)
        - [GitHub Portfolio Setup](https://example.com)
        - [Networking Strategies](https://example.com)
        - [Personal Branding Guide](https://example.com)
        """)
    
    with resource_cols[2]:
        st.markdown("""
        **📞 Support & Community**
        - [Help Center](https://example.com)
        - [Community Forum](https://example.com)
        - [Live Chat Support](https://example.com)
        - [Career Counseling](https://example.com)
        """)

    # Handle page redirects
    if hasattr(st.session_state, 'redirect_page') and st.session_state.redirect_page:
        # This would need to be handled by your main navigation logic
        pass
# ================== RESUME ANALYZER ==================
elif choice == "📂 Resume Analyzer":
    st.header("Upload Resume for Analysis")
    
    # Check if user is logged in
    if not st.session_state.user_email:
        st.warning("⚠ Please register first from the Home page to use this feature.")
        st.stop()

    if not st.session_state.subscribed and st.session_state.resume_uploads >= 1:
        st.warning("⚠ You have used your 1 free resume check. Please subscribe to continue.")
        if st.button("💎 Go to Subscription"):
            st.session_state.page = "💎 Subscription"
            st.rerun()
    else:
        # Custom CSS for File Uploader
        st.markdown("""
        <style>
        .upload-box {
            border: 2px dashed #4CAF50;
            border-radius: 12px;
            background-color: #f0fff4;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease-in-out;
        }
        .upload-box:hover {
            background-color: #e8f5e9;
            box-shadow: 0px 0px 12px rgba(76, 175, 80, 0.4);
            transform: scale(1.01);
        }
        </style>
        <div class="upload-box">
            <h4 style="color:#2e7d32; margin-bottom:10px;">📂 Drag & Drop your Resume here</h4>
        </div>
        """, unsafe_allow_html=True)

        # File Upload Widget
        uploaded_file = st.file_uploader("Upload Resume (pdf/docx/txt)", type=["pdf", "docx", "txt"])

        if uploaded_file is not None:
            resume_text = extract_text_from_uploaded_file(uploaded_file)
            if resume_text:
                st.text_area("Extracted Resume Text", resume_text, height=200)

                if st.button("Analyze with AI"):
                    with st.spinner("Analyzing your resume with Gemini..."):
                        time.sleep(2)

                        # Gemini prompt for ATS Score
                        score_prompt = f"""
                        You are an ATS evaluator. 
                        Analyze the following resume text and provide a single numeric ATS compatibility score out of 100. 
                        Only return the number without any explanation.
                        Resume:
                        {resume_text}
                        """
                        try:
                            model = genai.GenerativeModel("gemini-1.5-flash")
                            score_response = model.generate_content(score_prompt)
                            ats_score = int(re.findall(r"\d+", score_response.text.strip())[0])  
                        except Exception:
                            ats_score = 60  # fallback default

                        # Gemini prompt for detailed feedback
                        feedback = gemini_insights(f"Give detailed ATS-friendly feedback for this resume:\n{resume_text}")

                    # Save resume analysis to database
                    save_resume_analysis(
                        user_email=st.session_state.user_email,
                        resume_text=resume_text,
                        ats_score=ats_score,
                        feedback=feedback,
                        filename=uploaded_file.name
                    )

                    # Display Score
                    st.markdown(
                        f"""
                        <div style="display:flex; justify-content:center; align-items:center; margin:20px;">
                          <div style="
                              width:180px; height:180px;
                              border-radius:50%;
                              background:conic-gradient(#4CAF50 {ats_score*3.6}deg, #e0e0e0 0deg);
                              display:flex; flex-direction:column; justify-content:center; align-items:center;
                              font-size:26px; font-weight:bold; color:#4CAF50;">
                              {ats_score}%
                              <div style="font-size:14px; color:#333; font-weight:normal;">ATS Score</div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # AI Insights
                    st.subheader("AI Feedback")
                    st.info(feedback)

                    st.success("✅ Analysis saved to your profile!")

                    if not st.session_state.subscribed:
                        st.session_state.resume_uploads += 1
            else:
                st.error("Could not extract text from this file.")

# ================== JD MATCHER ==================
elif choice == "📄 JD Matcher":
    # Check if user is logged in
    if not st.session_state.user_email:
        st.warning("⚠ Please register first from the Home page to use this feature.")
        st.stop()

    # Initialize session state counter
    if "jd_uploads" not in st.session_state:
        st.session_state.jd_uploads = 0

    # Subscription check
    if not st.session_state.subscribed and st.session_state.jd_uploads >= 1:
        st.warning("⚠ You have used your 1 free JD match. Please subscribe to continue.")
        if st.button("💎 Go to Subscription"):
            st.session_state.page = "💎 Subscription"
            st.rerun()
    else:
        # File uploader and JD input
        resume_file = st.file_uploader("Upload your Resume", type=["pdf", "docx", "txt"])
        jd_text = st.text_area("Paste Job Description Here")

        # Process JD matching
        if resume_file is not None and jd_text:
            resume_text = extract_text_from_uploaded_file(resume_file)

            # Extract skills
            SKILLS_LIST = [
                "HTML", "CSS", "JavaScript", "Python", "Java", "C++", "C#", "SQL",
                "React", "Angular", "Node.js", "Django", "Flask", "AWS", "Azure",
                "Excel", "Power BI", "Tableau", "Git", "Docker", "Kubernetes",
                "Machine Learning", "Data Analysis", "TensorFlow", "PyTorch"
            ]

            def extract_skills(text):
                text_upper = text.upper()
                found_skills = [skill for skill in SKILLS_LIST if skill.upper() in text_upper]
                return found_skills

            resume_skills = extract_skills(resume_text)
            jd_skills = extract_skills(jd_text)
            missing_skills = [skill for skill in jd_skills if skill not in resume_skills]

            # Display Skills Boxes
            st.subheader("🔹 Resume Skills")
            resume_boxes = " ".join([
                f"<span style='background-color:#4CAF50;color:white;padding:5px 12px;margin:2px;border-radius:6px;font-weight:bold;'>{skill}</span>"
                for skill in resume_skills
            ])
            st.markdown(resume_boxes, unsafe_allow_html=True)

            st.subheader("🔹 Required Skills")
            jd_boxes = " ".join([
                f"<span style='background-color:#2196F3;color:white;padding:5px 12px;margin:2px;border-radius:6px;font-weight:bold;'>{skill}</span>"
                for skill in jd_skills
            ])
            st.markdown(jd_boxes, unsafe_allow_html=True)

            st.subheader("🔹 Missing Skills")
            missing_boxes = " ".join([
                f"<span style='background-color:#f44336;color:white;padding:5px 12px;margin:2px;border-radius:6px;font-weight:bold;'>{skill}</span>"
                for skill in missing_skills
            ])
            st.markdown(missing_boxes, unsafe_allow_html=True)

            # Weighted Similarity Score
            def match_resume_jd_weighted(resume_text, jd_text, skill_weight=0.7):
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity

                # Text similarity
                vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
                X = vect.fit_transform([jd_text, resume_text])
                text_sim = float(cosine_similarity(X[0], X[1])[0][0])

                # Skills similarity
                if jd_skills:
                    skills_match = len([s for s in jd_skills if s in resume_skills]) / len(jd_skills)
                else:
                    skills_match = 1.0

                weighted_sim = skill_weight * skills_match + (1-skill_weight) * text_sim
                return round(weighted_sim * 100, 1)

            sim_score = match_resume_jd_weighted(resume_text, jd_text)

            # Display Score
            st.subheader(f"Weighted Similarity Score: {sim_score}%")
            st.markdown(
                f"""
                <div style="display:flex; justify-content:center; align-items:center; margin:15px 0;">
                  <div style="
                      width:150px; height:150px;
                      border-radius:50%;
                      background: conic-gradient(#4CAF50 {sim_score*3.6}deg, #e0e0e0 0deg);
                      display:flex; flex-direction:column; justify-content:center; align-items:center;
                      font-size:28px; font-weight:bold; color:#4CAF50;">
                      {sim_score}%
                      <div style="font-size:14px; color:#333; font-weight:normal;">Weighted Score</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # AI Suggestions
            improvement_prompt = f"""
            You are an expert career coach. Analyze the following resume and job description. 
            Provide clear suggestions to improve the resume to better match the job description.
            
            Resume Text:
            {resume_text}
            
            Job Description:
            {jd_text}
            
            Focus on missing skills, keyword optimization, and ATS friendliness.
            """
            with st.spinner("Generating AI improvement suggestions..."):
                ai_feedback = gemini_insights(improvement_prompt)

            st.subheader("💡 AI Suggestions to Improve Resume")
            st.info(ai_feedback)

            # Save JD matching data to database
            save_jd_matching_data(
                user_email=st.session_state.user_email,
                similarity_score=sim_score,
                resume_skills=resume_skills,
                jd_skills=jd_skills,
                missing_skills=missing_skills,
                ai_suggestions=ai_feedback
            )

            st.success("✅ JD matching data saved to your profile!")

            # Increment JD uploads count
            if not st.session_state.subscribed:
                st.session_state.jd_uploads += 1

# ================== MASTERCLASS SECTION (Enhanced) ==================
elif choice == "🎓 Masterclass":
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .masterclass-hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px 20px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    .course-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 5px solid;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .course-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .course-card::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100px;
        background: linear-gradient(45deg, rgba(255,255,255,0.1), transparent);
        border-radius: 0 0 0 100px;
    }
    
    .data-science-card { border-left-color: #4CAF50; }
    .resume-card { border-left-color: #2196F3; }
    .interview-card { border-left-color: #FF9800; }
    .ml-card { border-left-color: #9C27B0; }
    .design-card { border-left-color: #F44336; }
    .business-card { border-left-color: #00BCD4; }
    
    .course-title {
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
    
    .course-mentor {
        color: #666;
        font-size: 1rem;
        margin-bottom: 8px;
    }
    
    .course-duration {
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 15px;
    }
    
    .course-badges {
        margin-bottom: 15px;
    }
    
    .badge {
        display: inline-block;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 8px;
        margin-bottom: 5px;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        background: linear-gradient(45deg, #f8f9fa, #e9ecef);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
        display: block;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
    
    .category-filter {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 20px 0;
    }
    
    .filter-btn {
        background: linear-gradient(45deg, #e3f2fd, #bbdefb);
        border: 2px solid transparent;
        border-radius: 25px;
        padding: 10px 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        color: #1976d2;
        font-weight: 500;
    }
    
    .filter-btn:hover {
        background: linear-gradient(45deg, #1976d2, #1565c0);
        color: white;
        transform: scale(1.05);
    }
    
    .ai-chat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 25px;
        margin-top: 30px;
        color: white;
    }
    
    .chat-input {
        background: rgba(255,255,255,0.1);
        border: 2px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        color: white;
        padding: 12px 15px;
    }
    
    .chat-input::placeholder {
        color: rgba(255,255,255,0.7);
    }
    
    .premium-banner {
        background: linear-gradient(45deg, #ffd700, #ffb300);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
        color: #333;
        box-shadow: 0 5px 15px rgba(255, 215, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="masterclass-hero">
        <div class="hero-title">🎓 Career Masterclasses</div>
        <div class="hero-subtitle">Learn from Industry Experts • Advance Your Career • Join 10,000+ Students</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Section
    st.markdown("""
    <div class="stats-container">
        <div class="stat-item">
            <span class="stat-number">50+</span>
            <span class="stat-label">Masterclasses</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">10K+</span>
            <span class="stat-label">Students Enrolled</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">98%</span>
            <span class="stat-label">Success Rate</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">4.9★</span>
            <span class="stat-label">Average Rating</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Category Filters
    st.markdown("### 🎯 Explore by Category")
    categories = ["All", "Data Science", "Resume & Career", "Interview Prep", "Tech Skills", "Design", "Business"]
    
    # Create filter buttons
    filter_cols = st.columns(len(categories))
    selected_category = "All"
    
    for idx, category in enumerate(categories):
        with filter_cols[idx]:
            if st.button(category, key=f"filter_{category}"):
                selected_category = category
    
    # Enhanced Course Data with more variety
    courses_data = [
        {
            "title": "Master Data Science from Scratch",
            "mentor": "Senior Data Scientist @ Google",
            "duration": "8 weeks • 32 hours",
            "category": "Data Science",
            "badges": ["Beginner Friendly", "Hands-on Projects", "Certificate"],
            "rating": "4.9",
            "students": "2.5K+",
            "link": "https://youtu.be/dMn2QFTyXUQ?si=SIKlONhrRiJIYrZk",
            "description": "Complete roadmap from Python basics to machine learning deployment",
            "card_class": "data-science-card"
        },
        {
            "title": "Build ATS-Winning Resumes",
            "mentor": "HR Director @ Microsoft",
            "duration": "4 weeks • 16 hours",
            "category": "Resume & Career",
            "badges": ["ATS Optimization", "Industry Templates", "1-on-1 Review"],
            "rating": "4.8",
            "students": "5.2K+",
            "link": "https://youtu.be/IIGWpw1FXhk?si=MS9FfmwqLWsMkA_k",
            "description": "Create resumes that pass ATS filters and impress recruiters",
            "card_class": "resume-card"
        },
        {
            "title": "Ace Technical Interviews",
            "mentor": "Engineering Manager @ Amazon",
            "duration": "6 weeks • 24 hours",
            "category": "Interview Prep",
            "badges": ["Live Mock Interviews", "System Design", "Coding Practice"],
            "rating": "4.9",
            "students": "3.8K+",
            "link": "https://youtu.be/vU3dL1cNqgQ?si=LLm5zA3HjGpTy1h2",
            "description": "Master coding interviews, system design, and behavioral questions",
            "card_class": "interview-card"
        },
        {
            "title": "Machine Learning Bootcamp",
            "mentor": "AI Research Scientist @ Tesla",
            "duration": "10 weeks • 40 hours",
            "category": "Tech Skills",
            "badges": ["Deep Learning", "Real Projects", "Industry Mentorship"],
            "rating": "4.9",
            "students": "1.9K+",
            "link": "https://youtu.be/example1",
            "description": "From basics to advanced ML algorithms and neural networks",
            "card_class": "ml-card"
        },
        {
            "title": "UI/UX Design Mastery",
            "mentor": "Design Lead @ Airbnb",
            "duration": "8 weeks • 32 hours",
            "category": "Design",
            "badges": ["Portfolio Projects", "Figma Expert", "User Research"],
            "rating": "4.8",
            "students": "2.7K+",
            "link": "https://youtu.be/example2",
            "description": "Create stunning user experiences with design thinking principles",
            "card_class": "design-card"
        },
        {
            "title": "Product Management Excellence",
            "mentor": "VP Product @ Stripe",
            "duration": "6 weeks • 24 hours",
            "category": "Business",
            "badges": ["Strategy Framework", "Case Studies", "Industry Network"],
            "rating": "4.8",
            "students": "1.5K+",
            "link": "https://youtu.be/example3",
            "description": "Learn product strategy, roadmapping, and stakeholder management",
            "card_class": "business-card"
        }
    ]
    
    # Filter courses based on selection
    if selected_category == "All":
        filtered_courses = courses_data
    else:
        filtered_courses = [course for course in courses_data if course["category"] == selected_category]
    
    # Display Premium Banner for non-subscribers
    if not st.session_state.subscribed:
        st.markdown("""
        <div class="premium-banner">
            <h3>🔓 Unlock All Masterclasses with Premium</h3>
            <p>Get access to 50+ expert-led courses, certificates, and 1-on-1 mentoring sessions</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Upgrade to Premium"):
            st.session_state.page = "💎 Subscription"
            st.rerun()
    
    # Course Cards
    st.markdown(f"### 📚 {selected_category} Courses ({len(filtered_courses)} available)")
    
    for course in filtered_courses:
        with st.container():
            st.markdown(f"""
            <div class="course-card {course['card_class']}">
                <div class="course-title">{course['title']}</div>
                <div class="course-mentor">👨‍🏫 {course['mentor']}</div>
                <div class="course-duration">⏱️ {course['duration']} | ⭐ {course['rating']} | 👥 {course['students']} enrolled</div>
                <div class="course-badges">
                    {''.join([f'<span class="badge">{badge}</span>' for badge in course['badges']])}
                </div>
                <p style="color: #666; line-height: 1.6;">{course['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Course Action Buttons
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if st.button(f"🎯 Enroll Now", key=f"enroll_{course['title']}"):
                    st.success(f"✅ Successfully enrolled in '{course['title']}'!")
                    st.balloons()
            with col2:
                if st.button("👀 Preview", key=f"preview_{course['title']}"):
                    st.info(f"🎬 Opening preview for '{course['title']}'...")
                    st.markdown(f"[🔗 Watch Preview]({course['link']})")
            with col3:
                if st.button("❤️ Wishlist", key=f"wishlist_{course['title']}"):
                    st.success("Added to wishlist!")
    
    # Learning Path Recommendations
    st.markdown("### 🛤️ Recommended Learning Paths")
    
    learning_paths = {
        "💼 Career Switcher to Tech": ["Build ATS-Winning Resumes", "Master Data Science from Scratch", "Ace Technical Interviews"],
        "🚀 Senior Professional Growth": ["Product Management Excellence", "Machine Learning Bootcamp", "UI/UX Design Mastery"],
        "🎯 Job Interview Ready": ["Build ATS-Winning Resumes", "Ace Technical Interviews", "Master Data Science from Scratch"]
    }
    
    for path_name, path_courses in learning_paths.items():
        with st.expander(f"{path_name} (3 courses)"):
            st.write("**Recommended sequence:**")
            for i, course_title in enumerate(path_courses, 1):
                st.write(f"{i}. {course_title}")
            if st.button(f"Start Learning Path", key=f"path_{path_name}"):
                st.success(f"🎉 Started learning path: {path_name}")
    
    # Enhanced AI Career Guidance
    st.markdown("""
    <div class="ai-chat-box">
        <h3 style="margin-top: 0; margin-bottom: 20px;">🤖 AI Career Advisor</h3>
        <p style="margin-bottom: 20px; opacity: 0.9;">Get personalized course recommendations and career guidance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Chat Interface
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input(
            "Ask your career question...", 
            placeholder="e.g., 'I want to transition from marketing to data science. What courses should I take?'",
            key="ai_career_chat"
        )
    with col2:
        ask_ai = st.button("🚀 Ask AI", key="ask_ai_btn")
    
    if user_query and ask_ai:
        with st.spinner("🧠 AI is thinking..."):
            # Enhanced career guidance prompt
            career_prompt = f"""
            You are an expert career advisor with knowledge of industry trends and skill requirements.
            Based on this question: "{user_query}"
            
            Provide:
            1. Specific course recommendations from our catalog
            2. Career roadmap with timeline
            3. Industry insights and salary expectations
            4. Skills to focus on
            5. Networking and job search tips
            
            Be encouraging, specific, and actionable.
            """
            
            ai_response = gemini_insights(career_prompt)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e8f5e9, #c8e6c9); 
                       border-radius: 15px; padding: 20px; margin: 15px 0;
                       border-left: 5px solid #4CAF50;">
                <h4 style="color: #2e7d32; margin-top: 0;">🎯 AI Career Advisor Says:</h4>
                <div style="color: #333; line-height: 1.6;">{ai_response}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick Action Buttons
    st.markdown("### ⚡ Quick Actions")
    action_cols = st.columns(4)
    
    with action_cols[0]:
        if st.button("📅 Browse Schedule"):
            st.info("📋 Upcoming sessions:\n• Data Science - Oct 15\n• Resume Workshop - Oct 18\n• Interview Prep - Oct 22")
    
    with action_cols[1]:
        if st.button("🏆 View Certificates"):
            st.success("🎓 Your earned certificates will appear here after course completion!")
    
    with action_cols[2]:
        if st.button("👥 Join Community"):
            st.info("💬 Join our Discord community with 5,000+ learners!")
    
    with action_cols[3]:
        if st.button("📊 Track Progress"):
            st.info("📈 Progress tracking available for enrolled students!")
# ================== SUBSCRIPTION SECTION ==================
elif choice == "💳 Subscription":
    st.markdown(
        """
        <style>
        .sub-header {
            font-size:26px;
            font-weight:600;
            color:#ffffff;
            text-align:center;
            margin-bottom:10px;
        }
        .info-box {
            background: linear-gradient(135deg, #4CAF50, #2E7D32);
            color: white;
            padding:20px;
            border-radius:15px;
            text-align:center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            margin-bottom:20px;
        }
        .info-text {
            font-size:16px;
            margin-top:5px;
        }
        .form-box {
            background-color:#f9f9f9;
            padding:20px;
            border-radius:12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            margin-bottom:20px;
        }
        .qr-box {
            text-align:center;
            margin-top:20px;
        }
        .small-btn button {
            width:200px !important;
            margin:auto;
            display:block;
        }
        .price-tag {
            text-align:center;
            font-size:16px;
            font-weight:600;
            color:#2E7D32;
            margin-top:8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Subscription header inside box
    st.markdown(
        """
        <div class="info-box">
            <div class="sub-header">💎 Premium Subscription</div>
            <div class="info-text">Get unlimited JD Matcher access 🚀<br>First Resume Free — Upgrade for more!</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Track form visibility
    if "show_form" not in st.session_state:
        st.session_state.show_form = False
    if "show_payment" not in st.session_state:
        st.session_state.show_payment = False

    # Smaller Buy Now button + price tag
    with st.container():
        st.markdown('<div class="small-btn">', unsafe_allow_html=True)
        if st.button("💳 Buy Now"):
            if st.session_state.user_email:
                st.session_state.show_payment = True
            else:
                st.session_state.show_form = True
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="price-tag">₹199/month • ₹999/year (Save 58%)</div>', unsafe_allow_html=True)

    # Subscription Form (for new users)
    if st.session_state.show_form and not st.session_state.user_email:
        st.markdown('<div class="form-box">', unsafe_allow_html=True)
        st.markdown("### 📝 Create Account First")
        with st.form("new_user_form", clear_on_submit=True):
            sub_name = st.text_input("👤 Full Name")
            sub_email = st.text_input("📧 Email Address")
            sub_phone = st.text_input("📱 Phone Number")
            password = st.text_input("🔒 Password", type="password")

            submitted = st.form_submit_button("✅ Create Account & Continue")
            if submitted:
                if sub_name and sub_email and sub_phone:
                    if save_user_basic_info(sub_name, sub_email, sub_phone):
                        st.success(f"Account created! Welcome {sub_name}!")
                        st.session_state.show_form = False
                        st.session_state.show_payment = True
                        st.rerun()
                else:
                    st.error("Please fill all details.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Payment Section (for logged in users)
    if st.session_state.show_payment and st.session_state.user_email:
        st.markdown('<div class="form-box">', unsafe_allow_html=True)
        st.markdown("### 💳 Choose Your Plan")
        
        # Plan Selection
        col1, col2 = st.columns(2)
        with col1:
            monthly_selected = st.button(
                """
                📅 *Monthly Plan*
                ₹199/month
                • Unlimited Resume Analysis
                • Unlimited JD Matching
                • Priority Support
                """,
                key="monthly_plan"
            )
        
        with col2:
            yearly_selected = st.button(
                """
                🏆 *Yearly Plan*
                ₹999/year
                • Everything in Monthly
                • Save ₹1389 per year
                • Exclusive Career Guidance
                • Priority Features Access
                """,
                key="yearly_plan"
            )
        
        if monthly_selected or yearly_selected:
            amount = 199 if monthly_selected else 999
            plan_type = "monthly" if monthly_selected else "yearly"
            
            # Create payment order
            order_id = create_payment_order(
                st.session_state.user_email, 
                st.session_state.user_name, 
                "", # Phone number can be fetched from user profile
                amount, 
                plan_type
            )
            
            if order_id:
                st.session_state.payment_order_id = order_id
                st.success(f"Order Created: {order_id}")
                
                # Real UPI QR Code Generation
                upi_id = "teenasaraswat04@oksbi"  # Your actual UPI ID
                merchant_name = "Resume Analyzer Pro"
                
                # UPI Payment URL format
                upi_url = f"upi://pay?pa={upi_id}&pn={merchant_name}&am={amount}&cu=INR&tn=Order-{order_id}"
                
                # Generate QR Code
                qr = qrcode.QRCode(version=1, box_size=10, border=5)
                qr.add_data(upi_url)
                qr.make(fit=True)
                
                qr_img = qr.make_image(fill_color="black", back_color="white")
                buf = io.BytesIO()
                qr_img.save(buf, format="PNG")
                buf.seek(0)
                
                # Display Payment Details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📱 Scan & Pay")
                    st.image(buf, caption=f"Pay ₹{amount} - Order: {order_id}", width=250)
                    
                    # Manual UPI Details
                    st.markdown(
                        f"""
                        *📋 Manual Payment Details:*
                        - *UPI ID:* {upi_id}
                        - *Amount:* ₹{amount}
                        - *Order ID:* {order_id}
                        
                        Please include Order ID in payment description
                        """)
                
                with col2:
                    st.markdown("### ⏱ Payment Status")
                    
                    # Payment verification section
                    if st.button("🔄 Check Payment Status"):
                        status = verify_payment_status(order_id)
                        if status == "completed":
                            st.success("✅ Payment Successful!")
                            st.balloons()
                        elif status == "pending":
                            st.warning("⏳ Payment Pending...")
                        else:
                            st.info("💡 Waiting for payment...")
                    
                    st.markdown("---")
                    
                    # Demo purpose - Manual completion button
                    st.markdown("🧪 For Demo Purpose:")
                    if st.button("✅ Mark Payment as Complete"):
                        if complete_payment(order_id):
                            st.success("🎉 Payment Successful! Premium activated!")
                            st.balloons()
                            st.session_state.show_payment = False
                            st.session_state.subscribed = True
                            st.rerun()
                    
                    # Payment Instructions
                    st.markdown(
                        """
                        *📝 Payment Instructions:*
                        1. Scan QR code with any UPI app
                        2. Enter amount: ₹{}
                        3. Add Order ID in remarks
                        4. Complete payment
                        5. Click "Check Status" above
                        """.format(amount))
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Alternative Payment Methods
    if st.session_state.show_payment:
        st.markdown("### 💳 Other Payment Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📱 PhonePe"):
                st.info("Redirecting to PhonePe... (Feature coming soon)")
        with col2:
            if st.button("💸 Paytm"):
                st.info("Redirecting to Paytm... (Feature coming soon)")
        with col3:
            if st.button("🏦 Net Banking"):
                st.info("Redirecting to Bank... (Feature coming soon)")

    # Remove old QR code section completely

# ================== PROFILE SECTION ==================
elif choice == "👤 Profile":
    if not st.session_state.user_email:
        st.warning("⚠ Please register first from the Home page to view your profile.")
        st.stop()
    
    st.header(f"👤 {st.session_state.user_name}'s Profile")
    
    user_profile = get_user_profile(st.session_state.user_email)
    
    if user_profile:
        user_info = user_profile["user_info"]
        
        # Profile Overview Cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 20px; border-radius: 12px; text-align: center;">
                    <h3 style="margin: 0; font-size: 24px;">{user_profile['resume_analyses']}</h3>
                    <p style="margin: 5px 0 0 0;">Resume Analyses</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                           color: white; padding: 20px; border-radius: 12px; text-align: center;">
                    <h3 style="margin: 0; font-size: 24px;">{user_profile['jd_matches']}</h3>
                    <p style="margin: 5px 0 0 0;">JD Matches</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            payment_count = len(user_profile['payments'])
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                           color: white; padding: 20px; border-radius: 12px; text-align: center;">
                    <h3 style="margin: 0; font-size: 24px;">{payment_count}</h3>
                    <p style="margin: 5px 0 0 0;">Payments Made</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # User Details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Account Details")
            st.write(f"*Name:* {user_info.get('name', 'N/A')}")
            st.write(f"*Email:* {user_info.get('email', 'N/A')}")
            st.write(f"*Phone:* {user_info.get('phone', 'N/A')}")
            st.write(f"*Joined:* {user_info.get('registration_date', 'N/A')}")
        
        with col2:
            st.subheader("💎 Subscription Status")
            status = user_info.get('subscription_status', 'free')
            if status == 'premium':
                expiry = user_info.get('subscription_expiry')
                st.success("🔓 *Premium Active*")
                if expiry:
                    st.write(f"*Expires:* {expiry.strftime('%d %b %Y')}")
            else:
                st.info("🔒 *Free Plan*")
                st.write("Upgrade to unlock unlimited features!")
        
        # Payment History
        if user_profile['payments']:
            st.subheader("💳 Payment History")
            for payment in user_profile['payments']:
                with st.expander(f"₹{payment.get('amount', 0)} - {payment.get('payment_date', 'N/A')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"*Amount:* ₹{payment.get('amount', 0)}")
                        st.write(f"*Method:* {payment.get('payment_method', 'N/A')}")
                        st.write(f"*Status:* ✅ {payment.get('status', 'N/A').title()}")
                    with col2:
                        st.write(f"*Plan:* {payment.get('subscription_type', 'N/A').title()}")
                        st.write(f"*Date:* {payment.get('payment_date', 'N/A')}")
        
        # Recent Activity
        st.subheader("📊 Recent Activity")
        
        resume_history, jd_history = get_user_history(st.session_state.user_email)
        
        if resume_history or jd_history:
            tab1, tab2 = st.tabs(["📂 Resume Analysis", "📄 JD Matching"])
            
            with tab1:
                if resume_history:
                    for i, record in enumerate(resume_history):
                        with st.expander(f"Analysis #{i+1} - Score: {record.get('ats_score', 'N/A')}%"):
                            st.write(f"*Date:* {record.get('analysis_date', 'N/A')}")
                            st.write(f"*File:* {record.get('filename', 'N/A')}")
                            st.write(f"*ATS Score:* {record.get('ats_score', 'N/A')}%")
                            if record.get('ai_feedback'):
                                st.write("*AI Feedback:*")
                                st.info(record['ai_feedback'][:200] + "..." if len(record['ai_feedback']) > 200 else record['ai_feedback'])
                else:
                    st.write("No resume analysis found. Upload a resume to get started!")
            
            with tab2:
                if jd_history:
                    for i, record in enumerate(jd_history):
                        with st.expander(f"Match #{i+1} - Score: {record.get('similarity_score', 'N/A')}%"):
                            st.write(f"*Date:* {record.get('matching_date', 'N/A')}")
                            st.write(f"*Similarity:* {record.get('similarity_score', 'N/A')}%")
                            
                            if record.get('missing_skills'):
                                st.write("*Missing Skills:*")
                                missing_skills_html = " ".join([
                                    f"<span style='background-color:#f44336;color:white;padding:3px 8px;margin:2px;border-radius:4px;font-size:12px;'>{skill}</span>"
                                    for skill in record['missing_skills']
                                ])
                                st.markdown(missing_skills_html, unsafe_allow_html=True)
                else:
                    st.write("No JD matching found. Try the JD Matcher tool!")
        else:
            st.info("Start using our tools to see your activity here!")

# ================== ABOUT US ==================
elif choice == "ℹ About Us":
    st.header("ℹ About Us")
    
    st.markdown(
        """
        <div style="background-color:#FFFDE7; padding:20px; border-radius:12px;">
            <h3 style="color: #FF9800;">📘 Project Overview</h3>
            <p>This platform is designed to help job seekers and professionals enhance their career prospects with AI-powered tools. It offers:</p>
            <ul>
                <li>✅ <strong>Resume Analyzer:</strong> Get instant feedback to make your resume ATS-friendly.</li>
                <li>✅ <strong>JD Matcher:</strong> Compare your resume with job descriptions to check keyword relevance and improve your application.</li>
                <li>✅ <strong>Career Masterclasses:</strong> Attend industry sessions and learn from experts on how to crack interviews and build professional skills.</li>
                <li>✅ <strong>Subscription Benefits:</strong> Unlock premium insights, unlimited resume checks, and priority career guidance.</li>
            </ul>
            <h3 style="color: #FF9800;">🚀 How It Works</h3>
            <p>The platform uses cutting-edge technologies like:</p>
            <ul>
                <li>💡 <strong>Google Gemini AI:</strong> Provides advanced career suggestions and personalized feedback.</li>
                <li>📊 <strong>TF-IDF & Similarity Matching:</strong> Matches your resume with job descriptions to highlight key skills.</li>
                <li>📂 <strong>File Processing:</strong> Extracts text from PDFs, DOCX, and TXT files for analysis.</li>
                <li>🗄 <strong>MongoDB Integration:</strong> Stores user data, resume analysis, and JD matching history.</li>
            </ul>
            <h3 style="color: #FF9800;">👥 Developed By</h3>
            <p>This project is built with passion by:</p>
            <ul>
                <li>👩‍💼 <strong>Teena Saraswat</strong></li>
                <li>👨‍💼 <strong>Prashant Sharma</strong></li>
            </ul>
            <p style="text-align:center; color: gray; font-style: italic;">Empowering careers, one resume at a time!</p>
        </div>
        """,
        unsafe_allow_html=True
    )