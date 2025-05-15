import pandas as pd
import spacy
import ast
import re
from experta import KnowledgeEngine, Fact, Rule, DefFacts, MATCH
from sklearn.utils import resample

# ------------------ Load Models & Data ------------------

nlp = spacy.load("en_core_web_sm")
df = pd.read_csv('Career Guidance Expert System.csv')  # Your dataset file

# ------------------ Balance Dataset ------------------

# Balance the dataset to avoid bias
def get_balanced_df(df, samples_per_class=119):
    balanced_frames = []
    for field, group in df.groupby('candidate_field'):
        if len(group) >= samples_per_class:
            sampled = group.sample(n=samples_per_class, random_state=42)
        else:
            sampled = resample(group, replace=True, n_samples=samples_per_class, random_state=42)
        balanced_frames.append(sampled)
    return pd.concat(balanced_frames).reset_index(drop=True)

balanced_df = get_balanced_df(df, samples_per_class=119)

# ------------------ Helper Functions ------------------

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

# Extract unique skills
unique_hard_skills = set()
unique_soft_skills = set()
for skills in balanced_df['hard_skill']:
    unique_hard_skills.update(parse_skills(skills))
for skills in balanced_df['soft_skill']:
    unique_soft_skills.update(parse_skills(skills))

hard_skill_dict = {skill.lower(): True for skill in unique_hard_skills}
soft_skill_dict = {skill.lower(): True for skill in unique_soft_skills}

print(f"✅ Loaded {len(hard_skill_dict)} hard skills and {len(soft_skill_dict)} soft skills from balanced data.")

# ------------------ Skill Extraction ------------------

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

    @DefFacts()
    def _initial_action(self):
        yield Fact(action="find_career_field")

    @Rule(Fact(action='find_career_field'), salience=100)
    def input_fact(self):
        if self.user_text:
            self.declare(Fact(text_input=self.user_text))

    @Rule(Fact(text_input=MATCH.text_input), salience=90)
    def analyze_text(self, text_input):
        print(f"\n📝 Analyzing user input: {text_input}")

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
            print(f"\n✅ Recommended Career Field Based on Your Skills: {best_career}")
            print(f"🔢 Raw Score: {self.career_scores[best_career]}")
            print(f"⚖️ Normalized Score: {normalized_scores[best_career]:.4f}")
        else:
            print("\n❌ Sorry, no matching career field found based on your skills.")

# ------------------ Main Program ------------------

if __name__ == "__main__":
    user_input = (
        "I enjoy planning promotional campaigns, managing social media content, and analyzing customer behavior. I'm creative, detail-oriented, and good at communication and branding."
    )

    hard_skills, soft_skills = extract_skills(user_input)

    print(f"\n🧠 Extracted Hard Skills: {hard_skills}")
    print(f"🧠 Extracted Soft Skills: {soft_skills}")

    engine = CareerExpertSystem(user_text=user_input)
    engine.reset()

    for hs in hard_skills:
        engine.declare(Fact(hard_skill=hs))
    for ss in soft_skills:
        engine.declare(Fact(soft_skill=ss))

    engine.run()
