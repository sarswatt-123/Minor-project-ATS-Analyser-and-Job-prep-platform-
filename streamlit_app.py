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
if choice == "🏠 Home":
    st.markdown(
        """
        <h2 style='text-align: center; color: #4CAF50;'> Welcome to the ATS Resume Platform 👋</h2>
        <p style='text-align: center; color: gray;'>Your one-stop solution for Resume Analysis, JD Matching, and Career Growth.</p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------- USER FORM -----------
    if not st.session_state.user_email:
        st.markdown(
            """
            <div style="background-color:#FFF3E0; padding:20px; border-radius:12px; margin-bottom:20px;">
                <h3 style="color:#E65100;">📝 Before You Continue</h3>
                <p style="color:gray;">Please enter your details to personalize your experience.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("user_details_form"):
            name = st.text_input("👤 Full Name")
            email = st.text_input("📧 Email Address")
            phone = st.text_input("📞 Phone Number")
            submitted = st.form_submit_button("✅ Submit & Continue")

        if submitted:
            if not name or not email or not phone:
                st.error("⚠ Please fill in all details before continuing.")
            else:
                if save_user_basic_info(name, email, phone):
                    st.success(f"Welcome, {name}! 🎉 You can now explore the platform.")
                    # Check subscription status
                    check_user_subscription(email)
                    st.rerun()
    else:
        user_profile = get_user_profile(st.session_state.user_email)
        if user_profile:
            st.success(f"Welcome back, {st.session_state.user_name}! You're logged in.")
            # Check subscription status
            check_user_subscription(st.session_state.user_email)

    # ----------- FEATURES CARDS -----------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div style="background-color:#E3F2FD; padding:20px; border-radius:12px; margin-bottom:15px;">
                <h4>📂 Resume Analyzer</h4>
                <p>Get instant ATS-friendly feedback on your resume to boost your job chances.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div style="background-color:#FFF3E0; padding:20px; border-radius:12px; margin-bottom:15px;">
                <h4>📄 JD Matcher</h4>
                <p>Compare your resume with a job description to check relevancy & keyword match.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div style="background-color:#E8F5E9; padding:20px; border-radius:12px; margin-bottom:15px;">
                <h4>🎓 Masterclass</h4>
                <p>Attend exclusive industry sessions and learn directly from professionals.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div style="background-color:#F3E5F5; padding:20px; border-radius:12px; margin-bottom:15px;">
                <h4>💎 Subscription</h4>
                <p>Unlock premium features, unlimited resume checks, and priority support.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

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

# ================== MASTERCLASS ==================
elif choice == "🎓 Masterclass":
    st.header("🎓 Career Masterclasses")
    st.write("Learn from industry experts. Explore our upcoming sessions:")

    courses = [
        {"title": "Crack Your First Data Analyst Job", "mentor": "Deloitte Expert", "link": "https://youtu.be/dMn2QFTyXUQ?si=SIKlONhrRiJIYrZk"},
        {"title": "How to Build ATS-Friendly Resume", "mentor": "Google Recruiter", "link": "https://youtu.be/IIGWpw1FXhk?si=MS9FfmwqLWsMkA_k"},
        {"title": "Ace Your Technical Interviews", "mentor": "Microsoft Engineer", "link": "https://youtu.be/vU3dL1cNqgQ?si=LLm5zA3HjGpTy1h2"}
    ]

    for course in courses:
        with st.expander(course["title"]):
            st.write(f"👨‍🏫 Mentor: {course['mentor']}")
            st.write(f"🔗 [Watch Here]({course['link']})")
            if st.button(f"Enroll: {course['title']}"):
                st.success("✅ You have enrolled successfully!")

    st.subheader("Ask AI about Career Guidance")
    user_q = st.text_input("Ask your career-related question:")
    if user_q:
        ai_answer = gemini_insights(user_q)
        st.info(ai_answer)

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