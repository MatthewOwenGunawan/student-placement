import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Placement Prediction Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_models():
    clf = joblib.load('artifacts/model_clf.pkl')
    reg = joblib.load('artifacts/model_reg.pkl')
    return clf, reg

try:
    clf_model, reg_model = load_models()
    models_ready = True
except Exception as e:
    st.error(f"Sistem gagal memuat model ML: {e}")
    models_ready = False

def main():
    st.markdown("<h2 style='text-align: center; color: #2C3E50;'>Student Placement & Salary Analytics</h2>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.markdown("### Profil Kandidat")
    
    with st.sidebar.form("input_form"):
        st.markdown("**1. Data Akademik**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        cgpa = st.number_input("CGPA (Skala 10)", 0.0, 10.0, 7.5, step=0.1)
        ssc_p = st.number_input("SSC Percentage (SMP)", 0.0, 100.0, 75.0)
        hsc_p = st.number_input("HSC Percentage (SMA)", 0.0, 100.0, 75.0)
        degree_p = st.number_input("Degree Percentage (S1/D3)", 0.0, 100.0, 75.0)
        
        st.markdown("**2. Evaluasi Kompetensi**")
        tech_score = st.slider("Technical Skill Score", 0, 100, 80)
        soft_score = st.slider("Soft Skill Score", 0, 100, 80)
        entrance_score = st.number_input("Entrance Exam Score", 0.0, 100.0, 75.0)
        
        st.markdown("**3. Pengalaman & Aktivitas**")
        internship = st.number_input("Total Internship", 0, 5, 1)
        projects = st.number_input("Total Live Projects", 0, 10, 1)
        work_ex = st.number_input("Work Experience (Bulan)", 0, 60, 0)
        certifications = st.number_input("Sertifikasi Profesional", 0, 10, 1)
        extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        
        st.markdown("**4. Kedisiplinan**")
        attendance = st.number_input("Attendance Percentage (%)", 0.0, 100.0, 85.0)
        backlogs = st.number_input("History Backlogs", 0, 10, 0)
        
        st.markdown("---")
        st.markdown("**Konfigurasi Sistem HRD**")
        threshold = st.slider("Ambang Batas Kelulusan (%)", 10, 90, 50, help="Turunkan angka ini agar model lebih toleran/mudah meluluskan kandidat.") / 100.0
        
        submit_button = st.form_submit_button("Jalankan Analisis")

    if submit_button and models_ready:
        data = {
            "gender": gender,
            "ssc_percentage": ssc_p,
            "hsc_percentage": hsc_p,
            "degree_percentage": degree_p,
            "cgpa": cgpa,
            "entrance_exam_score": entrance_score,
            "technical_skill_score": tech_score,
            "soft_skill_score": soft_score,
            "internship_count": internship,
            "live_projects": projects,
            "work_experience_months": work_ex,
            "certifications": certifications,
            "attendance_percentage": attendance,
            "backlogs": backlogs,
            "extracurricular_activities": extra
        }
        df = pd.DataFrame([data])

        df['avg_skill'] = (df['technical_skill_score'] + df['soft_skill_score']) / 2
        df['activity_score'] = df['internship_count'] + df['certifications'] + df['live_projects']
        df['has_experience'] = (df['work_experience_months'] > 0).astype(int)
        df['has_backlog'] = (df['backlogs'] > 0).astype(int)

     
        try:
            probabilities = clf_model.predict_proba(df)[0]
            peluang_lulus = probabilities[1] 
            
            is_placed = bool(peluang_lulus >= threshold)
            
            prob_text = f"Peluang Kesuksesan Kandidat: **{peluang_lulus*100:.1f}%**"
            
        except:
            raw_status = clf_model.predict(df)[0]
            is_placed = (str(raw_status) == "1" or str(raw_status).lower() == "placed")
            prob_text = "Peluang Kesuksesan: Terkalkulasi secara absolut."

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("#### Hasil Evaluasi Model")
            st.info(prob_text) 
            
            if is_placed:
                st.success("STATUS: MEMENUHI SYARAT PENEMPATAN (PLACED)")
                
                base_salary = float(reg_model.predict(df)[0])
                bonus_cgpa = (cgpa - 7.5) * 0.5      
                bonus_tech = (tech_score - 80) * 0.05 
                bonus_exp = (internship * 0.2) + (projects * 0.1)
                
                final_salary = base_salary + bonus_cgpa + bonus_tech + bonus_exp
                final_salary = max(3.0, final_salary) # Batas bawah aman
                
                st.info(f"Estimasi Kompensasi (Salary Package): **{final_salary:.2f} LPA**")
                
            else:
                st.error("STATUS: TIDAK MEMENUHI SYARAT PENEMPATAN (NOT PLACED)")
                st.warning("Estimasi Kompensasi: Tidak tersedia.")
                st.markdown(f"*Catatan:* Kandidat ditolak karena Peluang sukses ({peluang_lulus*100:.1f}%) berada di bawah ambang batas yang ditetapkan ({threshold*100:.1f}%).")

        with col2:
            st.markdown("#### Analisis Distribusi Kompetensi")
            categories = ['Tech Skill', 'Soft Skill', 'Entrance Exam', 'Attendance', 'Degree %']
            values = [tech_score, soft_score, entrance_score, attendance, degree_p]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                fillcolor='rgba(41, 128, 185, 0.3)',
                line=dict(color='#2980b9', width=2),
                name='Profil'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color='#bdc3c7')
                ),
                showlegend=False,
                margin=dict(l=30, r=30, t=30, b=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Silakan lengkapi parameter di sidebar dan jalankan analisis.")

if __name__ == "__main__":
    main()