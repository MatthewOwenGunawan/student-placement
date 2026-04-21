import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Placement Prediction Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.markdown(
        "<h2 style='text-align: center; color: #2C3E50;'>Student Placement & Salary Analytics</h2>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.sidebar.markdown("### Profil Kandidat")

    with st.sidebar.form("input_form"):

        gender = st.selectbox("Gender", ["Male", "Female"])
        cgpa = st.number_input("CGPA (Skala 10)", 0.0, 10.0, 7.5, step=0.1)

        ssc_p = st.number_input("SSC Percentage", 0.0, 100.0, 75.0)
        hsc_p = st.number_input("HSC Percentage", 0.0, 100.0, 75.0)
        degree_p = st.number_input("Degree Percentage", 0.0, 100.0, 75.0)

        tech_score = st.slider("Technical Skill Score", 0, 100, 80)
        soft_score = st.slider("Soft Skill Score", 0, 100, 80)
        entrance_score = st.number_input("Entrance Exam Score", 0.0, 100.0, 75.0)

        internship = st.number_input("Internship Count", 0, 5, 1)
        projects = st.number_input("Live Projects", 0, 10, 1)
        work_ex = st.number_input("Work Experience (Months)", 0, 60, 0)
        certifications = st.number_input("Certifications", 0, 10, 1)

        attendance = st.number_input("Attendance (%)", 0.0, 100.0, 85.0)
        backlogs = st.number_input("Backlogs", 0, 10, 0)

        extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])

        threshold = st.slider(
            "Ambang Batas Kelulusan (%)",
            10, 90, 50
        ) / 100.0

        submit_button = st.form_submit_button("Jalankan Analisis")

    if submit_button and models_ready:

        df = pd.DataFrame([{
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
        }])

        df["avg_skill"] = (df["technical_skill_score"] + df["soft_skill_score"]) / 2

        df["activity_score"] = (
            df["internship_count"] +
            df["certifications"] +
            df["live_projects"]
        )

        df["has_experience"] = (df["work_experience_months"] > 0).astype(int)
        df["has_backlog"] = (df["backlogs"] > 0).astype(int)

        try:
            probs = clf_model.predict_proba(df)[0]
            peluang_lulus = probs[1]
        except Exception as e:
            st.error(f"Error predict classification: {e}")
            return

        is_placed = peluang_lulus >= threshold

        prob_text = f"Peluang Placement: **{peluang_lulus*100:.1f}%**"

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("### Hasil Evaluasi Model")
            st.info(prob_text)

            if is_placed:
                st.success("STATUS: PLACED")

                try:
                    base_salary = float(reg_model.predict(df)[0])
                except Exception as e:
                    st.error(f"Error predict regression: {e}")
                    base_salary = 0

                bonus = (
                    (cgpa - 7.5) * 0.5 +
                    (tech_score - 80) * 0.05 +
                    (internship * 0.2) +
                    (projects * 0.1)
                )

                final_salary = max(3.0, base_salary + bonus)

                st.info(f"Estimasi Salary: **{final_salary:.2f} LPA**")

            else:
                st.error("STATUS: NOT PLACED")
                st.warning("Tidak ada estimasi salary.")

        with col2:
            st.markdown("### Radar Kompetensi")

            categories = [
                'Tech Skill',
                'Soft Skill',
                'Entrance Exam',
                'Attendance',
                'Degree %'
            ]

            values = [
                tech_score,
                soft_score,
                entrance_score,
                attendance,
                degree_p
            ]

            N = len(categories)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

            values += values[:1]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

            ax.plot(angles, values, linewidth=2)
            ax.fill(angles, values, alpha=0.25)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)

            ax.set_ylim(0, 100)

            st.pyplot(fig)

    else:
        st.info("Silakan input data")


if __name__ == "__main__":
    main()