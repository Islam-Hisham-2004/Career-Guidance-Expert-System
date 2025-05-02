import pandas as pd
import spacy
import ast
import re
from experta import KnowledgeEngine, Fact, Rule
import types

# ------------------ Load Models & Data ------------------

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv(
    r'Career Guidance Expert System.csv')


# ---------- Helper Functions ----------

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

for skills in df['hard_skill']:
    unique_hard_skills.update(parse_skills(skills))
for skills in df['soft_skill']:
    unique_soft_skills.update(parse_skills(skills))

hard_skill_dict = {skill.lower(): True for skill in unique_hard_skills}
soft_skill_dict = {skill.lower(): True for skill in unique_soft_skills}

print(f"Loaded {len(hard_skill_dict)} hard skills and {len(soft_skill_dict)} soft skills.")


# ---------- Skill Extraction ----------

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


# ---------- Expert System ----------

class CareerExpertSystem(KnowledgeEngine):
    pass  # rules will be added dynamically


engine = CareerExpertSystem()


def create_rule(hard_skills_row, soft_skills_row, candidate_field):
    # Create conditions
    condition_list = []
    for hs in hard_skills_row:
        condition_list.append(f"Fact(hard_skill='{hs.lower()}')")
    for ss in soft_skills_row:
        condition_list.append(f"Fact(soft_skill='{ss.lower()}')")

    conditions = ', '.join(condition_list)

    # Define rule function body
    rule_code = f"""
def _rule(self):
    print("\\nRecommended field based on your skills: {candidate_field}")
"""

    exec(rule_code, globals(), locals())
    rule_func = locals()['_rule']

    # Attach rule to engine dynamically
    rule_decorator = Rule(eval(conditions))
    setattr(engine.__class__, f"recommend_{candidate_field.replace(' ', '_')}_{len(engine.facts)}",
            rule_decorator(rule_func))


# ---------- Dynamically Add Rules ----------

for _, row in df[df['label'] == 1].iterrows():
    hard_skills_row = parse_skills(row['hard_skill'])
    soft_skills_row = parse_skills(row['soft_skill'])
    create_rule(hard_skills_row, soft_skills_row, row['candidate_field'])

print(f"Generated {len(engine.__class__.__dict__) - 1} rules from dataset.")  # exclude __module__

# ---------- Example Input ----------

user_input = "I have experience in nursing, registration, and service. I am also good at written communication."

hard_skills, soft_skills = extract_skills(user_input)

print("Extracted Hard Skills:", hard_skills)
print("Extracted Soft Skills:", soft_skills)

# ---------- Run Expert System ----------

# Declare facts
for hs in hard_skills:
    engine.declare(Fact(hard_skill=hs))
for ss in soft_skills:
    engine.declare(Fact(soft_skill=ss))

engine.run()
