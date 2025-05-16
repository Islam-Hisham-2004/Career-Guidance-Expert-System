import streamlit as st
import pandas as pd
import ast
import spacy
import re
from experta import KnowledgeEngine, Fact, Rule, DefFacts, MATCH
from sklearn.utils import resample
import time

# ------------------ Load NLP and Data ------------------

nlp = spacy.load("en_core_web_sm")

@st.cache_data
def load_data():
    df = pd.read_csv("Career Guidance Expert System.csv")

    def get_balanced_df(df, samples_per_class=119):
        balanced_frames = []
        for field, group in df.groupby('candidate_field'):
            if len(group) >= samples_per_class:
                sampled = group.sample(n=samples_per_class, random_state=42)
            else:
                sampled = resample(group, replace=True, n_samples=samples_per_class, random_state=42)
            balanced_frames.append(sampled)
        return pd.concat(balanced_frames).reset_index(drop=True)

    return get_balanced_df(df)

balanced_df = load_data()

# ------------------ Skill Extraction ------------------

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

unique_hard_skills = set()
unique_soft_skills = set()
for skills in balanced_df['hard_skill']:
    unique_hard_skills.update(parse_skills(skills))
for skills in balanced_df['soft_skill']:
    unique_soft_skills.update(parse_skills(skills))

hard_skill_dict = {skill.lower(): True for skill in unique_hard_skills}
soft_skill_dict = {skill.lower(): True for skill in unique_soft_skills}

def match_skill_in_text(skill, text):
    pattern = r'\b' + re.escape(skill.lower()) + r'\b'
    return re.search(pattern, text.lower()) is not None

def extract_skills(user_input):
    tokens, lemmas = process_text_spacy(user_input)
    hard_skills = set()
    soft_skills = set()

    for skill in hard_skill_dict:
        if match_skill_in_text(skill, user_input):
            hard_skills.add(skill)
    for skill in soft_skill_dict:
        if match_skill_in_text(skill, user_input):
            soft_skills.add(skill)

    for lemma in lemmas:
        if lemma in hard_skill_dict:
            hard_skills.add(lemma)
        if lemma in soft_skill_dict:
            soft_skills.add(lemma)

    return list(hard_skills), list(soft_skills)

# ------------------ Expert System ------------------

class CareerExpertSystem(KnowledgeEngine):
    def __init__(self, user_text=None):
        super().__init__()
        self.user_text = user_text
        self.career_scores = {}
        self.field_counts = balanced_df['candidate_field'].value_counts().to_dict()
        self.result = ""

    @DefFacts()
    def _initial_action(self):
        yield Fact(action="find_career_field")

    @Rule(Fact(action='find_career_field'), salience=100)
    def input_fact(self):
        if self.user_text:
            self.declare(Fact(text_input=self.user_text))

    @Rule(Fact(text_input=MATCH.text_input), salience=90)
    def analyze_text(self, text_input):
        pass

    @Rule(Fact(hard_skill=MATCH.skill))
    def match_career_by_hard_skill(self, skill):
        matching_rows = balanced_df[balanced_df['hard_skill'].str.lower().str.contains(skill, na=False)]
        for _, row in matching_rows.iterrows():
            career = row['candidate_field']
            self.career_scores[career] = self.career_scores.get(career, 0) + 1

    @Rule(Fact(soft_skill=MATCH.skill))
    def match_career_by_soft_skill(self, skill):
        matching_rows = balanced_df[balanced_df['soft_skill'].str.lower().str.contains(skill, na=False)]
        for _, row in matching_rows.iterrows():
            career = row['candidate_field']
            self.career_scores[career] = self.career_scores.get(career, 0) + 1

    @Rule(Fact(action='find_career_field'), salience=-10)
    def recommend_best_career(self):
        if self.career_scores:
            normalized_scores = {
                field: score / self.field_counts.get(field, 1)
                for field, score in self.career_scores.items()
            }
            best_career = max(normalized_scores, key=normalized_scores.get)

            self.result = (
                f"### 🎯 Recommended Career Field\n"
                f"##  {best_career}"
            )

            with st.expander("📊 More details"):
                st.write(f"🔢 Raw Score: {self.career_scores[best_career]}  _(Matching rows)_")
                st.write(f"⚖️ Normalized Score: {normalized_scores[best_career]:.4f}  _(Adjusted by field size)_")
                st.write("💼 Hard Skills Matched:", self.user_hard_skills)
                st.write("🧠 Soft Skills Matched:", self.user_soft_skills)
        else:
            self.result = "❌ No matching career field found."

# ------------------ Streamlit GUI ------------------

st.title("Career Guidance Expert System")

user_input = st.text_area(f"## 📝 Describe your skills and interests:")

status = st.empty()
result_container = st.empty()

if st.button("Find"):
    if user_input.strip():
        hard_skills, soft_skills = extract_skills(user_input)

        engine = CareerExpertSystem(user_text=user_input)
        engine.user_hard_skills = hard_skills
        engine.user_soft_skills = soft_skills

        status.write("⏳ Analyzing...")
        time.sleep(1)

        engine.reset()
        for hs in hard_skills:
            engine.declare(Fact(hard_skill=hs))
        for ss in soft_skills:
            engine.declare(Fact(soft_skill=ss))
        engine.run()

        status.empty()
        result_container.markdown(engine.result)
    else:
        st.warning("⚠️ Please enter your input first.")
