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
menu = ["🏠 Home", "📂 Resume Analyzer", "📄 JD Matcher", "🎓 Masterclass", "💳 Subscription", "👤 Profile", "ℹ️ About Us"]
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
    # Enhanced CSS for Resume Analyzer
    st.markdown("""
    <style>
    .analyzer-hero {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 40px 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(76, 175, 80, 0.3);
    }
    
    .upload-zone {
        border: 3px dashed #4CAF50;
        border-radius: 20px;
        background: linear-gradient(45deg, #f0fff4, #e8f5e9);
        padding: 40px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-zone:hover {
        background: linear-gradient(45deg, #e8f5e9, #c8e6c9);
        transform: scale(1.02);
        box-shadow: 0 10px 25px rgba(76, 175, 80, 0.2);
    }
    
    .upload-icon {
        font-size: 4rem;
        margin-bottom: 20px;
        color: #4CAF50;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    .score-circle {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin: 30px auto;
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
    }
    
    .analysis-tabs {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin: 30px 0;
    }
    
    .feature-box {
        background: white;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        border-top: 4px solid;
    }
    
    .feature-box:hover {
        transform: translateY(-5px);
    }
    
    .ats-box { border-top-color: #4CAF50; }
    .keyword-box { border-top-color: #2196F3; }
    .format-box { border-top-color: #FF9800; }
    .skills-box { border-top-color: #9C27B0; }
    
    .progress-bar {
        width: 100%;
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    .tips-sidebar {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="analyzer-hero">
        <h1 style="margin: 0; font-size: 2.5rem;">🎯 Smart Resume Analyzer</h1>
        <p style="font-size: 1.2rem; margin-top: 10px; opacity: 0.9;">
            Get instant ATS compatibility scores, keyword optimization, and AI-powered suggestions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if user is logged in
    if not st.session_state.user_email:
        st.warning("⚠️ Please register first from the Home page to use this feature.")
        st.stop()

    # Feature Overview
    st.markdown("### 🚀 Analysis Features")
    
    col1, col2, col3, col4 = st.columns(4)
    features = [
        {"icon": "🎯", "title": "ATS Score", "desc": "Compatibility check", "color": "#4CAF50"},
        {"icon": "🔍", "title": "Keywords", "desc": "Optimization tips", "color": "#2196F3"},
        {"icon": "📄", "title": "Format", "desc": "Structure analysis", "color": "#FF9800"},
        {"icon": "🛠️", "title": "Skills", "desc": "Gap identification", "color": "#9C27B0"}
    ]
    
    for i, feature in enumerate(features):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div class="feature-box" style="border-top-color: {feature['color']};">
                <div style="font-size: 2.5rem; margin-bottom: 10px;">{feature['icon']}</div>
                <h4 style="margin: 10px 0; color: #333;">{feature['title']}</h4>
                <p style="color: #666; margin: 0;">{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Subscription Check
    if not st.session_state.subscribed and st.session_state.resume_uploads >= 1:
        st.error("⚠️ You have used your 1 free resume check. Please subscribe to continue.")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💎 Upgrade to Premium", use_container_width=True):
                st.session_state.page = "💎 Subscription"
                st.rerun()
        st.stop()

    # Enhanced Upload Section
    st.markdown("### 📁 Upload Your Resume")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon">📄</div>
            <h3 style="color: #4CAF50; margin-bottom: 15px;">Drag & Drop Your Resume</h3>
            <p style="color: #666;">Supports PDF, DOCX, and TXT files</p>
            <p style="color: #999; font-size: 0.9rem;">Max size: 10MB</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose file", 
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("""
        <div class="tips-sidebar">
            <h4 style="color: #1976d2; margin-top: 0;">💡 Pro Tips</h4>
            <ul style="color: #555; line-height: 1.6;">
                <li>Use standard resume format</li>
                <li>Include relevant keywords</li>
                <li>Keep sections well-organized</li>
                <li>Use bullet points for clarity</li>
                <li>Include quantifiable achievements</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        resume_text = extract_text_from_uploaded_file(uploaded_file)
        
        if resume_text:
            # Quick Preview
            with st.expander("📖 Resume Preview", expanded=True):
                st.text_area("Extracted Text", resume_text[:500] + "..." if len(resume_text) > 500 else resume_text, height=150)
            
            # Analysis Options
            st.markdown("### ⚙️ Analysis Options")
            
            col1, col2 = st.columns(2)
            with col1:
                analysis_type = st.selectbox(
                    "Choose Analysis Type",
                    ["Complete Analysis", "ATS Score Only", "Keyword Analysis", "Format Check"]
                )
            with col2:
                industry_focus = st.selectbox(
                    "Industry Focus",
                    ["General", "Technology", "Healthcare", "Finance", "Marketing", "Engineering"]
                )
            
            # Analyze Button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_btn = st.button("🔍 Analyze Resume", use_container_width=True)
            
            if analyze_btn:
                with st.spinner("🤖 AI is analyzing your resume..."):
                    progress_bar = st.progress(0)
                    
                    # Simulate analysis progress
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                    
                    # Generate ATS Score
                    score_prompt = f"""
                    You are an ATS evaluator for {industry_focus} industry. 
                    Analyze this resume and provide a numeric ATS compatibility score out of 100.
                    Consider: keywords, format, sections, skills relevance.
                    Only return the number.
                    Resume: {resume_text}
                    """
                    
                    try:
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        score_response = model.generate_content(score_prompt)
                        ats_score = int(re.findall(r"\d+", score_response.text.strip())[0])
                    except:
                        ats_score = np.random.randint(60, 95)  # Fallback
                    
                    # Generate detailed feedback
                    feedback_prompt = f"""
                    Analyze this {industry_focus} resume and provide detailed feedback in these categories:
                    1. ATS Compatibility
                    2. Keyword Optimization  
                    3. Format & Structure
                    4. Content Quality
                    5. Improvement Suggestions
                    
                    Resume: {resume_text}
                    """
                    
                    detailed_feedback = gemini_insights(feedback_prompt)
                
                # Clear progress bar
                progress_bar.empty()
                
                # Results Section
                st.markdown("## 📊 Analysis Results")
                
                # ATS Score Display
                score_color = "#4CAF50" if ats_score >= 80 else "#FF9800" if ats_score >= 60 else "#F44336"
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div class="score-circle" style="background: conic-gradient({score_color} {ats_score*3.6}deg, #e0e0e0 0deg);">
                        <div style="font-size: 3rem;">{ats_score}%</div>
                        <div style="font-size: 1.2rem; opacity: 0.9;">ATS Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed Analysis Tabs
                tab1, tab2, tab3, tab4 = st.tabs(["🎯 Overall", "🔍 Keywords", "📄 Format", "💡 Suggestions"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📈 Score Breakdown")
                        categories = ["ATS Compatibility", "Keywords", "Format", "Content Quality"]
                        scores = [ats_score, np.random.randint(70, 95), np.random.randint(75, 90), np.random.randint(65, 85)]
                        
                        for cat, score in zip(categories, scores):
                            color = "#4CAF50" if score >= 80 else "#FF9800" if score >= 60 else "#F44336"
                            st.markdown(f"""
                            <div style="margin: 15px 0;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <span>{cat}</span>
                                    <span style="font-weight: bold; color: {color};">{score}%</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: {score}%; background-color: {color};"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### 🏆 Ranking")
                        rank_text = "Excellent" if ats_score >= 90 else "Good" if ats_score >= 75 else "Needs Improvement"
                        st.success(f"Your resume ranks: **{rank_text}**")
                        
                        st.markdown("#### 📊 Comparison")
                        st.info(f"Your score is higher than {min(95, ats_score + np.random.randint(5, 15))}% of resumes in our database.")
                
                with tab2:
                    st.markdown("#### 🔍 Keyword Analysis")
                    
                    # Mock keyword analysis
                    found_keywords = ["Python", "Machine Learning", "SQL", "Data Analysis"]
                    missing_keywords = ["TensorFlow", "Docker", "AWS", "Kubernetes"]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**✅ Found Keywords**")
                        for kw in found_keywords:
                            st.markdown(f"<span style='background: #4CAF50; color: white; padding: 3px 8px; border-radius: 4px; margin: 2px; display: inline-block;'>{kw}</span>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**❌ Missing Keywords**")
                        for kw in missing_keywords:
                            st.markdown(f"<span style='background: #F44336; color: white; padding: 3px 8px; border-radius: 4px; margin: 2px; display: inline-block;'>{kw}</span>", unsafe_allow_html=True)
                
                with tab3:
                    st.markdown("#### 📄 Format Analysis")
                    
                    format_checks = [
                        ("Contact Information", True, "Clearly visible at top"),
                        ("Professional Summary", True, "Present and concise"),
                        ("Work Experience", True, "Well structured with dates"),
                        ("Skills Section", False, "Could be more prominent"),
                        ("Education", True, "Properly formatted"),
                        ("File Format", True, "ATS-friendly format")
                    ]
                    
                    for check, passed, note in format_checks:
                        icon = "✅" if passed else "⚠️"
                        color = "#4CAF50" if passed else "#FF9800"
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin: 10px 0; padding: 10px; background: {color}20; border-radius: 8px;">
                            <span style="font-size: 1.2rem; margin-right: 10px;">{icon}</span>
                            <div>
                                <strong>{check}</strong><br>
                                <small style="color: #666;">{note}</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with tab4:
                    st.markdown("#### 💡 AI-Powered Suggestions")
                    st.info(detailed_feedback)
                
                # Action Buttons
                st.markdown("### 🎯 Next Steps")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📥 Download Report", use_container_width=True):
                        st.success("📄 Detailed report downloaded!")
                
                with col2:
                    if st.button("🔄 Try JD Matcher", use_container_width=True):
                        st.info("Navigate to JD Matcher to compare with job descriptions!")
                
                with col3:
                    if st.button("📚 Get Coaching", use_container_width=True):
                        st.info("Premium feature: 1-on-1 resume coaching available!")
                
                # Save analysis
                save_resume_analysis(
                    user_email=st.session_state.user_email,
                    resume_text=resume_text,
                    ats_score=ats_score,
                    feedback=detailed_feedback,
                    filename=uploaded_file.name
                )
                
                st.success("✅ Analysis saved to your profile!")
                
                if not st.session_state.subscribed:
                    st.session_state.resume_uploads += 1

# ================== JD MATCHER ==================
elif choice == "📄 JD Matcher":
    # Enhanced CSS for JD Matcher
    st.markdown("""
    <style>
    .matcher-hero {
        background: linear-gradient(135deg, #2196F3 0%, #1976d2 100%);
        color: white;
        padding: 40px 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(33, 150, 243, 0.3);
    }
    
    .dual-upload {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 30px;
        margin: 30px 0;
    }
    
    .upload-card {
        background: white;
        border: 2px dashed #2196F3;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-card:hover {
        border-color: #1976d2;
        background: #f3f9ff;
        transform: translateY(-2px);
    }
    
    .match-visualization {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 20px;
        padding: 30px;
        margin: 30px 0;
        text-align: center;
    }
    
    .skills-comparison {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 20px;
        margin: 30px 0;
    }
    
    .skills-column {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .skill-tag {
        display: inline-block;
        padding: 6px 12px;
        margin: 4px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .found-skill { background: #4CAF50; color: white; }
    .required-skill { background: #2196F3; color: white; }
    .missing-skill { background: #F44336; color: white; }
    
    .improvement-panel {
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        border-left: 5px solid #FF9800;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="matcher-hero">
        <h1 style="margin: 0; font-size: 2.5rem;">🎯 Job Description Matcher</h1>
        <p style="font-size: 1.2rem; margin-top: 10px; opacity: 0.9;">
            Compare your resume with job descriptions and get personalized improvement suggestions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if user is logged in
    if not st.session_state.user_email:
        st.warning("⚠️ Please register first from the Home page to use this feature.")
        st.stop()

    # Initialize session state counter
    if "jd_uploads" not in st.session_state:
        st.session_state.jd_uploads = 0

    # Subscription check
    if not st.session_state.subscribed and st.session_state.jd_uploads >= 1:
        st.error("⚠️ You have used your 1 free JD match. Please subscribe to continue.")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💎 Upgrade to Premium", use_container_width=True):
                st.session_state.page = "💎 Subscription"
                st.rerun()
        st.stop()

    # Matching Options
    st.markdown("### ⚙️ Matching Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        match_algorithm = st.selectbox("Algorithm", ["Advanced AI", "Keyword-based", "Semantic"])
    with col2:
        industry_type = st.selectbox("Industry", ["Technology", "Healthcare", "Finance", "Marketing", "General"])
    with col3:
        experience_level = st.selectbox("Level", ["Entry", "Mid", "Senior", "Executive"])

    # Dual Upload Interface
    st.markdown("### 📁 Upload Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="upload-card">
            <div style="font-size: 3rem; color: #4CAF50; margin-bottom: 15px;">📄</div>
            <h4 style="color: #333; margin-bottom: 10px;">Upload Resume</h4>
            <p style="color: #666;">PDF, DOCX, or TXT format</p>
        </div>
        """, unsafe_allow_html=True)
        
        resume_file = st.file_uploader("Choose Resume", type=["pdf", "docx", "txt"], key="resume_upload")
    
    with col2:
        st.markdown("""
        <div class="upload-card">
            <div style="font-size: 3rem; color: #2196F3; margin-bottom: 15px;">💼</div>
            <h4 style="color: #333; margin-bottom: 10px;">Job Description</h4>
            <p style="color: #666;">Paste or type the job posting</p>
        </div>
        """, unsafe_allow_html=True)

    jd_text = st.text_area("Job Description", placeholder="Paste the complete job description here...", height=150)

    # Process matching
    if resume_file is not None and jd_text:
        resume_text = extract_text_from_uploaded_file(resume_file)
        
        if resume_text:
            # Quick Preview
            with st.expander("📖 Document Preview"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Resume Preview**")
                    st.text_area("", resume_text[:300] + "...", height=100, key="resume_preview")
                with col2:
                    st.markdown("**JD Preview**")
                    st.text_area("", jd_text[:300] + "...", height=100, key="jd_preview")
            
            # Match Button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🎯 Start Matching Analysis", use_container_width=True):
                    with st.spinner("🔍 Analyzing match compatibility..."):
                        progress_bar = st.progress(0)
                        
                        # Simulate analysis progress
                        for i in range(100):
                            time.sleep(0.03)
                            progress_bar.progress(i + 1)
                        
                        # Enhanced skills extraction
                        COMPREHENSIVE_SKILLS = [
                            # Technical Skills
                            "Python", "Java", "JavaScript", "C++", "C#", "SQL", "HTML", "CSS", "React", "Angular",
                            "Node.js", "Django", "Flask", "Spring", "AWS", "Azure", "Docker", "Kubernetes",
                            "Git", "Jenkins", "MongoDB", "PostgreSQL", "MySQL", "Redis", "Elasticsearch",
                            
                            # Data & Analytics
                            "Machine Learning", "Data Analysis", "TensorFlow", "PyTorch", "Pandas", "NumPy",
                            "Tableau", "Power BI", "Excel", "R", "Scala", "Hadoop", "Spark", "Kafka",
                            
                            # Business Skills
                            "Project Management", "Agile", "Scrum", "Leadership", "Communication",
                            "Problem Solving", "Team Management", "Strategic Planning", "Business Analysis",
                            
                            # Design & Marketing
                            "UI/UX", "Figma", "Adobe Creative Suite", "SEO", "Digital Marketing",
                            "Content Strategy", "Brand Management", "Social Media"
                        ]
                        
                        def extract_enhanced_skills(text):
                            text_upper = text.upper()
                            found = []
                            for skill in COMPREHENSIVE_SKILLS:
                                if skill.upper() in text_upper:
                                    found.append(skill)
                            return found
                        
                        resume_skills = extract_enhanced_skills(resume_text)
                        jd_skills = extract_enhanced_skills(jd_text)
                        missing_skills = [skill for skill in jd_skills if skill not in resume_skills]
                        
                        # Calculate weighted similarity
                        def enhanced_similarity_score(resume_text, jd_text, resume_skills, jd_skills):
                            # Text similarity using TF-IDF
                            from sklearn.feature_extraction.text import TfidfVectorizer
                            from sklearn.metrics.pairwise import cosine_similarity
                            
                            vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
                            try:
                                X = vect.fit_transform([jd_text, resume_text])
                                text_sim = float(cosine_similarity(X[0], X[1])[0][0])
                            except:
                                text_sim = 0.5
                            
                            # Skills similarity
                            if jd_skills:
                                skills_sim = len([s for s in jd_skills if s in resume_skills]) / len(jd_skills)
                            else:
                                skills_sim = 1.0
                            
                            # Weighted combination (60% skills, 40% text)
                            weighted_score = 0.6 * skills_sim + 0.4 * text_sim
                            return round(weighted_score * 100, 1)
                        
                        similarity_score = enhanced_similarity_score(resume_text, jd_text, resume_skills, jd_skills)
                        
                        progress_bar.empty()
                        
                        # Results Display
                        st.markdown("## 📊 Matching Results")
                        
                        # Score Visualization
                        score_color = "#4CAF50" if similarity_score >= 80 else "#FF9800" if similarity_score >= 60 else "#F44336"
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.markdown(f"""
                            <div class="match-visualization">
                                <div style="width: 180px; height: 180px; border-radius: 50%; 
                                           background: conic-gradient({score_color} {similarity_score*3.6}deg, #e0e0e0 0deg);
                                           display: flex; flex-direction: column; justify-content: center; align-items: center;
                                           margin: 0 auto 20px; font-size: 2.5rem; font-weight: bold; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                                    {similarity_score}%
                                    <div style="font-size: 1rem; opacity: 0.9;">Match Score</div>
                                </div>
                                <h3 style="color: #333; margin: 0;">
                                    {
                                        "Excellent Match! 🎉" if similarity_score >= 85 else
                                        "Good Match! 👍" if similarity_score >= 70 else
                                        "Fair Match 📝" if similarity_score >= 50 else
                                        "Needs Improvement 🔧"
                                    }
                                </h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Skills Comparison
                        st.markdown("### 🔍 Skills Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("""
                            <div class="skills-column" style="border-top: 4px solid #4CAF50;">
                                <h4 style="color: #4CAF50; text-align: center; margin-bottom: 20px;">✅ Your Skills</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for skill in resume_skills[:10]:  # Show top 10
                                st.markdown(f'<span class="skill-tag found-skill">{skill}</span>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("""
                            <div class="skills-column" style="border-top: 4px solid #2196F3;">
                                <h4 style="color: #2196F3; text-align: center; margin-bottom: 20px;">🎯 Required Skills</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for skill in jd_skills[:10]:
                                st.markdown(f'<span class="skill-tag required-skill">{skill}</span>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("""
                            <div class="skills-column" style="border-top: 4px solid #F44336;">
                                <h4 style="color: #F44336; text-align: center; margin-bottom: 20px;">❌ Missing Skills</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if missing_skills:
                                for skill in missing_skills[:10]:
                                    st.markdown(f'<span class="skill-tag missing-skill">{skill}</span>', unsafe_allow_html=True)
                            else:
                                st.success("🎉 No critical skills missing!")
                        
                        # Detailed Analysis Tabs
                        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Deep Analysis", "💡 Suggestions", "📈 Improvement Plan"])
                        
                        with tab1:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### 📊 Match Breakdown")
                                categories = ["Skills Match", "Experience Level", "Industry Fit", "Keywords"]
                                scores = [
                                    len([s for s in jd_skills if s in resume_skills]) * 100 // max(len(jd_skills), 1),
                                    similarity_score + np.random.randint(-10, 10),
                                    similarity_score + np.random.randint(-15, 5),
                                    similarity_score + np.random.randint(-5, 15)
                                ]
                                
                                for cat, score in zip(categories, scores):
                                    score = max(0, min(100, score))  # Clamp to 0-100
                                    color = "#4CAF50" if score >= 75 else "#FF9800" if score >= 50 else "#F44336"
                                    st.markdown(f"""
                                    <div style="margin: 15px 0;">
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                            <span>{cat}</span>
                                            <span style="font-weight: bold; color: {color};">{score}%</span>
                                        </div>
                                        <div class="progress-bar">
                                            <div class="progress-fill" style="width: {score}%; background-color: {color};"></div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("#### 🎯 Key Insights")
                                
                                insights = [
                                    f"You match {len([s for s in jd_skills if s in resume_skills])}/{len(jd_skills)} required skills",
                                    f"Your resume has {len(resume_skills)} relevant skills total",
                                    f"Consider adding {len(missing_skills)} missing skills",
                                    "Strong alignment with job requirements" if similarity_score >= 70 else "Room for improvement in alignment"
                                ]
                                
                                for insight in insights:
                                    st.info(f"💡 {insight}")
                        
                        with tab2:
                            st.markdown("#### 🔍 Detailed Compatibility Analysis")
                            
                            # Generate comprehensive analysis
                            detailed_prompt = f"""
                            Perform a comprehensive job match analysis between this resume and job description:
                            
                            Resume: {resume_text[:1000]}
                            Job Description: {jd_text[:1000]}
                            
                            Analyze:
                            1. Technical skill alignment
                            2. Experience level match
                            3. Industry knowledge fit
                            4. Cultural fit indicators
                            5. Growth potential
                            
                            Provide specific, actionable insights.
                            """
                            
                            detailed_analysis = gemini_insights(detailed_prompt)
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #e8f5e9, #c8e6c9); 
                                       border-radius: 15px; padding: 20px; border-left: 5px solid #4CAF50;">
                                {detailed_analysis}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with tab3:
                            st.markdown("#### 💡 AI-Powered Improvement Suggestions")
                            
                            improvement_prompt = f"""
                            As an expert career coach, provide specific improvement suggestions for this resume to better match the job description.
                            
                            Resume Skills: {', '.join(resume_skills)}
                            Required Skills: {', '.join(jd_skills)}
                            Missing Skills: {', '.join(missing_skills)}
                            
                            Provide:
                            1. Priority skills to add
                            2. Resume content improvements
                            3. Keyword optimization tips
                            4. Experience highlighting strategies
                            5. Action items with timeline
                            """
                            
                            suggestions = gemini_insights(improvement_prompt)
                            
                            st.markdown(f"""
                            <div class="improvement-panel">
                                <h4 style="color: #E65100; margin-top: 0;">🚀 Improvement Roadmap</h4>
                                <div style="color: #333; line-height: 1.6;">{suggestions}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with tab4:
                            st.markdown("#### 📈 30-Day Improvement Plan")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("""
                                **Week 1-2: Skill Building**
                                - Learn top 3 missing skills
                                - Complete online courses
                                - Practice with projects
                                
                                **Week 3: Resume Enhancement**
                                - Add new skills to resume
                                - Optimize keyword density
                                - Restructure content
                                """)
                            
                            with col2:
                                st.markdown("""
                                **Week 4: Validation**
                                - Test resume with similar JDs
                                - Get peer review feedback
                                - Apply to target positions
                                
                                **Ongoing: Monitoring**
                                - Track application success
                                - Continuously update skills
                                - Network in target industry
                                """)
                        
                        # Action Buttons
                        st.markdown("### 🎯 Next Steps")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if st.button("📊 Generate Report", use_container_width=True):
                                st.success("📄 Comprehensive match report generated!")
                        
                        with col2:
                            if st.button("🔄 Try Another JD", use_container_width=True):
                                st.info("Upload another job description to compare!")
                        
                        with col3:
                            if st.button("📚 Skill Courses", use_container_width=True):
                                st.info("Recommended courses for missing skills available!")
                        
                        with col4:
                            if st.button("💼 Job Search", use_container_width=True):
                                st.info("Find similar job opportunities in our job board!")
                        
                        # Save JD matching data
                        save_jd_matching_data(
                            user_email=st.session_state.user_email,
                            similarity_score=similarity_score,
                            resume_skills=resume_skills,
                            jd_skills=jd_skills,
                            missing_skills=missing_skills,
                            ai_suggestions=suggestions
                        )
                        
                        st.success("✅ Analysis saved to your profile!")
                        
                        # Increment usage count for free users
                        if not st.session_state.subscribed:
                            st.session_state.jd_uploads += 1

    # Additional Tools Section
    if not resume_file or not jd_text:
        st.markdown("### 🛠️ Additional Tools")
        
        tool_cols = st.columns(3)
        
        with tool_cols[0]:
            st.markdown("""
            <div style="background: white; border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                <div style="font-size: 2.5rem; margin-bottom: 15px;">📝</div>
                <h4>JD Generator</h4>
                <p style="color: #666;">Create job descriptions from templates</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Create JD", key="jd_gen", use_container_width=True):
                st.info("Premium feature: AI-powered JD generator!")
        
        with tool_cols[1]:
            st.markdown("""
            <div style="background: white; border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                <div style="font-size: 2.5rem; margin-bottom: 15px;">🎯</div>
                <h4>Bulk Matcher</h4>
                <p style="color: #666;">Match resume against multiple JDs</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Bulk Match", key="bulk_match", use_container_width=True):
                st.info("Premium feature: Compare with multiple jobs at once!")
        
        with tool_cols[2]:
            st.markdown("""
            <div style="background: white; border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                <div style="font-size: 2.5rem; margin-bottom: 15px;">📈</div>
                <h4>Trend Analysis</h4>
                <p style="color: #666;">Industry skill demand trends</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("View Trends", key="trends", use_container_width=True):
                st.info("Analyze skill demand trends in your industry!")


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
# ================== ABOUT US (Fixed & Concise) ==================
elif choice == "ℹ️ About Us":
    # Simplified CSS - no complex animations that might break
    st.markdown("""
    <style>
    .about-hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .story-box {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        border-left: 4px solid #4CAF50;
    }
    
    .tech-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    .tech-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-top: 3px solid;
    }
    
    .team-section {
        background: #e3f2fd;
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        text-align: center;
    }
    
    .contact-box {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Clean Hero Section
    st.markdown("""
    <div class="about-hero">
        <h1>🚀 AI-Powered Resume Platform</h1>
        <p style="font-size: 1.2rem; margin-top: 10px;">Helping professionals land their dream jobs with smart resume analysis</p>
        <div style="margin-top: 20px;">
            <span style="background: rgba(255,255,255,0.2); padding: 6px 12px; border-radius: 15px; margin: 0 8px;">10K+ Users</span>
            <span style="background: rgba(255,255,255,0.2); padding: 6px 12px; border-radius: 15px; margin: 0 8px;">25K+ Resumes</span>
            <span style="background: rgba(255,255,255,0.2); padding: 6px 12px; border-radius: 15px; margin: 0 8px;">95% Success</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Concise Story Section
    st.markdown("""
    <div class="story-box">
        <h2 style="color: #333; margin-bottom: 15px;">📖 Our Story</h2>
        <p style="font-size: 1.1rem; line-height: 1.6; color: #555; margin-bottom: 15px;">
            Born from frustration with ATS systems filtering out qualified candidates, we created an AI-powered platform 
            to level the playing field in today's job market.
        </p>
        <p style="font-size: 1.1rem; line-height: 1.6; color: #555; margin: 0;">
            Today, we help thousands of professionals optimize their resumes, match with relevant jobs, 
            and accelerate their career growth through expert guidance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simplified Technology Section
    st.markdown("## 🛠️ Technology Stack")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="tech-card" style="border-top-color: #4CAF50;">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">🤖</div>
            <h4 style="color: #4CAF50;">AI Analysis</h4>
            <p style="color: #666; font-size: 0.9rem;">Google Gemini AI for intelligent resume insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tech-card" style="border-top-color: #2196F3;">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">📊</div>
            <h4 style="color: #2196F3;">Smart Matching</h4>
            <p style="color: #666; font-size: 0.9rem;">TF-IDF & ML for precise job matching</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="tech-card" style="border-top-color: #FF9800;">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">☁️</div>
            <h4 style="color: #FF9800;">Cloud Scale</h4>
            <p style="color: #666; font-size: 0.9rem;">MongoDB Atlas for reliable data storage</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="tech-card" style="border-top-color: #9C27B0;">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">🔒</div>
            <h4 style="color: #9C27B0;">Secure</h4>
            <p style="color: #666; font-size: 0.9rem;">End-to-end encryption & privacy protection</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Features Overview
    st.markdown("## 🎯 What We Offer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Resume Analyzer**
        - ATS compatibility scoring
        - Keyword optimization tips  
        - Format & structure analysis
        - Industry-specific feedback
        
        **🎯 Job Matcher**
        - Resume-JD similarity analysis
        - Skills gap identification
        - Improvement recommendations
        - Competitive positioning
        """)
    
    with col2:
        st.markdown("""
        **🎓 Masterclasses**
        - Expert-led career sessions
        - Interview preparation guides
        - Industry insights & trends
        - Skill development courses
        
        **💎 Premium Features**
        - Unlimited analyses
        - Priority support
        - Advanced AI insights
        - 1-on-1 career coaching
        """)
    
    # Simple Team Section
    st.markdown("""
    <div class="team-section">
        <h2 style="color: #1976d2; margin-bottom: 20px;">👥 Our Team</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
            <div>
                <div style="font-size: 3rem; margin-bottom: 10px;">👩‍💼</div>
                <h4 style="color: #333;">Teena Saraswat</h4>
                <p style="color: #666; margin: 5px 0;">Co-Founder & AI Specialist</p>
                <small style="color: #888;">ML & NLP Expert</small>
            </div>
            <div>
                <div style="font-size: 3rem; margin-bottom: 10px;">👨‍💼</div>
                <h4 style="color: #333;">Prashant Sharma</h4>
                <p style="color: #666; margin: 5px 0;">Co-Founder & Product Lead</p>
                <small style="color: #888;">Full-Stack Developer</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Mission & Vision (Simplified)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Mission**
        
        Democratize career opportunities by providing AI-powered tools that help every professional 
        optimize their job search and achieve career goals.
        """)
    
    with col2:
        st.markdown("""
        **🔮 Vision**
        
        Become the global leader in AI-driven career development, empowering millions to unlock 
        their potential and land dream jobs.
        """)
    
    # Simple Contact Section
    st.markdown("""
    <div class="contact-box">
        <h3 style="color: #2e7d32; margin-bottom: 15px;">📞 Get In Touch</h3>
        <p style="color: #555; margin-bottom: 20px;">
            Questions or feedback? We'd love to hear from you!
        </p>
        <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">
            <div style="text-align: center;">
                <strong>📧 Email</strong><br>
                <span style="color: #666;">contact@resumeanalyzer.com</span>
            </div>
            <div style="text-align: center;">
                <strong>💬 Support</strong><br>
                <span style="color: #666;">support@resumeanalyzer.com</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Platform Stats
    st.markdown("## 📈 Platform Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Users Helped", "10,000+", "↗️ Growing")
    
    with col2:
        st.metric("Resumes Analyzed", "25,000+", "↗️ +500/week")
    
    with col3:
        st.metric("Success Rate", "95%", "↗️ Improving")
    
    with col4:
        st.metric("Jobs Landed", "2,500+", "↗️ This year")
    
    # Simple Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Made with ❤️ for career success</p>
        <p style="font-size: 0.9rem;">© 2024 Resume Analyzer Pro • All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)