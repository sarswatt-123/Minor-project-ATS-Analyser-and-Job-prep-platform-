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

MONGO_URI = "mongodb+srv://teena3:123@cluster0.ojomaf6.mongodb.net/" # connection string 


# Connect to MongoDB
client = MongoClient(MONGO_URI)

# Database aur Collection choose karo
db = client["myDatabase"]        # apne database ka naam rakh sakti ho
collection = db["user_data"]     # collection ka naam (table jaisa hota hai)


# ================== CONFIG ==================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="AI Resume + Job Prep Tool", layout="wide")
st.title("🚀 AI-Powered Resume + Job Prep Platform")

# ================== SESSION STATE ==================
if "resume_uploads" not in st.session_state:
    st.session_state.resume_uploads = 0
if "subscribed" not in st.session_state:
    st.session_state.subscribed = False

# ================== HELPERS ==================
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
menu = ["🏠 Home", "📂 Resume Analyzer", "📄 JD Matcher", "🎓 Masterclass", "💳 Subscription", "ℹ About Us"]
choice = st.sidebar.selectbox("Navigate", menu)

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
            try:
                user_data = {"name": name, "email": email, "phone": phone}
                collection.insert_one(user_data)
                st.success(f"Welcome, {name}! 🎉 You can now explore the platform.")
            except Exception as e:
                st.error(f"❌ Database Error: {e}")




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

    if not st.session_state.subscribed and st.session_state.resume_uploads >= 1:
        st.warning("⚠ You have used your 1 free resume check. Please subscribe to continue.")
        if st.button("💎 Go to Subscription"):
            st.session_state.page = "💎 Subscription"
            st.experimental_rerun()
    else:
        uploaded_file = st.file_uploader("Upload Resume (pdf/docx/txt)", type=["pdf", "docx", "txt"])
        if uploaded_file is not None:
            resume_text = extract_text_from_uploaded_file(uploaded_file)
            if resume_text:
                st.text_area("Extracted Resume Text", resume_text, height=200)

                if st.button("Analyze with AI"):
                    with st.spinner("Analyzing your resume..."):
                        time.sleep(2)
                        feedback = gemini_insights(f"Give ATS-friendly feedback for this resume:\n{resume_text}")
                    st.subheader("AI Feedback")
                    st.info(feedback)

                    if not st.session_state.subscribed:
                        st.session_state.resume_uploads += 1
            else:
                st.error("Could not extract text from this file.")

# ================== JD MATCHER ==================
elif choice == "📄 JD Matcher":
    st.header("Resume vs Job Description Matcher")

    resume_file = st.file_uploader("Upload your Resume", type=["pdf", "docx", "txt"])
    jd_text = st.text_area("Paste Job Description Here")

    if resume_file is not None and jd_text:
        resume_text = extract_text_from_uploaded_file(resume_file)
        if st.button("Match Resume with JD"):
            sim, top_terms, present, missing = match_resume_jd_tfidf(resume_text, jd_text)
            st.subheader(f"Similarity Score: {sim}%")
            st.write("✅ Present in Resume:", present)
            st.write("❌ Missing from Resume:", missing)

# ================== MASTERCLASS ==================
elif choice == "🎓 Masterclass":
    st.header("🎓 Career Masterclasses")
    st.write("Learn from industry experts. Explore our upcoming sessions:")

    courses = [
        {"title": "Crack Your First Data Analyst Job", "mentor": "Deloitte Expert", "link": "https://youtu.be/example1"},
        {"title": "How to Build ATS-Friendly Resume", "mentor": "Google Recruiter", "link": "https://youtu.be/example2"},
        {"title": "Ace Your Technical Interviews", "mentor": "Microsoft Engineer", "link": "https://youtu.be/example3"}
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
# ================== SUBSCRIPTION SECTION ==================
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
            <div class="info-text">Get unlimited JD Matcher access 🚀<br>First Resume Free – Upgrade for more!</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Track form visibility
    if "show_form" not in st.session_state:
        st.session_state.show_form = False

    # Smaller Buy Now button + price tag
    with st.container():
        st.markdown('<div class="small-btn">', unsafe_allow_html=True)
        if st.button("💳 Buy Now"):
            st.session_state.show_form = True
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="price-tag">₹199/month • Lifetime Access Available</div>', unsafe_allow_html=True)

    # Subscription Form
    if st.session_state.show_form:
        st.markdown('<div class="form-box">', unsafe_allow_html=True)
        with st.form("subscription_form", clear_on_submit=True):
            name = st.text_input("👤 Full Name")
            email = st.text_input("📧 Email Address")
            password = st.text_input("🔑 Password", type="password")
            phone = st.text_input("📱 Phone Number")

            submitted = st.form_submit_button("✅ Continue")
            if submitted:
                st.success(f"Thank you {name}! Your details are saved ✅. Now you can proceed with payment.")
                st.session_state.show_form = False
        st.markdown('</div>', unsafe_allow_html=True)

    # QR Code Box
    st.markdown('<div class="qr-box">', unsafe_allow_html=True)
    st.write("### 📌 Scan & Pay")

    payment_link = "https://your-payment-gateway.com/pay?amount=199"
    qr = qrcode.make(payment_link)
    buf = io.BytesIO()
    qr.save(buf, format="PNG")
    buf.seek(0)

    st.image(buf, caption="Scan to Pay ₹199", width=250)
    st.markdown('</div>', unsafe_allow_html=True)




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