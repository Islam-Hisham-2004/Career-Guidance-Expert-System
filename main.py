import streamlit as st
import pandas as pd
import ast
import spacy
import re
from experta import KnowledgeEngine, Fact, Rule, DefFacts, MATCH
from sklearn.utils import resample
import time

# ---------- Page Setup & Style ----------
st.set_page_config(page_title="Career Guidance Expert System", layout="centered")

# Darker input box
st.markdown("""
    <style>
    textarea {
        background-color: #e6e8eb !important;
        color: #0a1e3c !important;
        border: 1px solid #b0b3b8 !important;
        border-radius: 6px !important;
        padding: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- NLP & Data ----------
nlp = spacy.load("en_core_web_sm")

@st.cache_data
def load_data():
    df = pd.read_csv("Career Guidance Expert System.csv")
    def get_balanced_df(df, samples_per_class=119):
        balanced = []
        for field, group in df.groupby('candidate_field'):
            if len(group) >= samples_per_class:
                sampled = group.sample(n=samples_per_class, random_state=42)
            else:
                sampled = resample(group, replace=True, n_samples=samples_per_class, random_state=42)
            balanced.append(sampled)
        return pd.concat(balanced).reset_index(drop=True)
    return get_balanced_df(df)

balanced_df = load_data()

# ---------- Skill Matching ----------
def process_text_spacy(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    return tokens, lemmas

def parse_skills(skill_str):
    try:
        return ast.literal_eval(skill_str)
    except:
        return [s.strip() for s in skill_str.split(',')]

# Extract all skills from dataset
hard_skill_dict = {s.lower(): True for skills in balanced_df['hard_skill'] for s in parse_skills(skills)}
soft_skill_dict = {s.lower(): True for skills in balanced_df['soft_skill'] for s in parse_skills(skills)}

def match_skill_in_text(skill, text):
    return re.search(r'\b' + re.escape(skill.lower()) + r'\b', text.lower()) is not None

def extract_skills(user_input):
    tokens, lemmas = process_text_spacy(user_input)
    hard_skills = {s for s in hard_skill_dict if match_skill_in_text(s, user_input) or s in lemmas}
    soft_skills = {s for s in soft_skill_dict if match_skill_in_text(s, user_input) or s in lemmas}
    return list(hard_skills), list(soft_skills)

# ---------- Expert System ----------
class CareerExpertSystem(KnowledgeEngine):
    def __init__(self, user_text=None):
        super().__init__()
        self.user_text = user_text
        self.career_scores = {}
        self.field_counts = balanced_df['candidate_field'].value_counts().to_dict()
        self.result = None

    @DefFacts()
    def _initial_action(self):
        yield Fact(action="find_career_field")

    @Rule(Fact(action='find_career_field'), salience=100)
    def input_fact(self):
        if self.user_text:
            self.declare(Fact(text_input=self.user_text))

    @Rule(Fact(hard_skill=MATCH.skill))
    def match_by_hard(self, skill):
        rows = balanced_df[balanced_df['hard_skill'].str.lower().str.contains(skill, na=False)]
        for _, row in rows.iterrows():
            self.career_scores[row['candidate_field']] = self.career_scores.get(row['candidate_field'], 0) + 1

    @Rule(Fact(soft_skill=MATCH.skill))
    def match_by_soft(self, skill):
        rows = balanced_df[balanced_df['soft_skill'].str.lower().str.contains(skill, na=False)]
        for _, row in rows.iterrows():
            self.career_scores[row['candidate_field']] = self.career_scores.get(row['candidate_field'], 0) + 1

    @Rule(Fact(action='find_career_field'), salience=-10)
    def recommend(self):
        if self.career_scores:
            norm_scores = {
                field: score / self.field_counts.get(field, 1)
                for field, score in self.career_scores.items()
            }
            best = max(norm_scores, key=norm_scores.get)
            self.result = f"### 🎯 Recommended Career Field\n## ✅ {best}"
            with st.expander("📊 More details"):
                st.write(f"🔢 Raw Score: {self.career_scores[best]}  *(Number of times your skills matched profiles in this career field)*")
                st.write(f"⚖️ Normalized Score: {norm_scores[best]:.4f}  *(This adjusts for fields with more or fewer total examples to ensure fairness)*")
                st.write("💼 Hard Skills Matched:", self.user_hard_skills)
                st.write("🧠 Soft Skills Matched:", self.user_soft_skills)
        else:
            self.result = None

# ---------- UI ----------
st.markdown("<h1 style='text-align:center;'> Career Guidance Expert System</h1>", unsafe_allow_html=True)
st.markdown("#### 📝 Describe your skills and interests:")

user_input = st.text_area("", height=150, placeholder="e.g. I enjoy solving technical problems and helping others...")

status = st.empty()
result_container = st.empty()

col_center = st.columns([1, 2, 1])[1]
with col_center:
    if st.button("🔍 Find", use_container_width=True):
        if user_input.strip():
            hard_skills, soft_skills = extract_skills(user_input)
            engine = CareerExpertSystem(user_text=user_input)
            engine.user_hard_skills = hard_skills
            engine.user_soft_skills = soft_skills

            status.info("⏳ Analyzing your input, please wait...")
            time.sleep(1)

            engine.reset()
            for hs in hard_skills:
                engine.declare(Fact(hard_skill=hs))
            for ss in soft_skills:
                engine.declare(Fact(soft_skill=ss))
            engine.run()

            status.empty()

            if engine.result:
                result_container.success(engine.result)
            else:
                result_container.error("❌ No matching career field found.")
        else:
            st.warning("⚠️ Please enter your input first.")
