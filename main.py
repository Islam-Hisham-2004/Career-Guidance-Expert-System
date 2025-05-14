import spacy
import re
from experta import *

# ------------------ Load Models & Data ------------------
nlp = spacy.load("en_core_web_sm")

# Define the career fields and their matching data
career_fields = {
    "Retail & Consumer Products": {
        "required_skills": ["customer service", "sales", "inventory management", "merchandising", "product knowledge", "retail software", "negotiation", "marketing"],
        "matching_keywords": ["store", "retail", "consumer products", "shopping", "merchandise", "customer service", "inventory"]
    },
    "Sales": {
        "required_skills": ["sales strategy", "negotiation", "customer relationship management (CRM)", "communication", "product knowledge", "lead generation", "closing deals"],
        "matching_keywords": ["sales", "selling", "negotiation", "CRM", "market research", "lead generation", "sales strategy"]
    },
    "Healthcare & Medical": {
        "required_skills": ["patient care", "nursing", "medical knowledge", "first aid", "medication administration", "clinical skills", "hospital protocols", "healthcare technology", "diagnosis", "medical ethics"],
        "matching_keywords": ["nurse", "doctor", "healthcare", "patient care", "hospital", "clinic", "medicine", "nursing", "health"]
    },
    "Telecommunication": {
        "required_skills": ["network administration", "telecom infrastructure", "customer service", "telecommunication systems", "problem-solving", "project management"],
        "matching_keywords": ["telecommunication", "network", "cellular", "wireless", "internet service", "call center", "data transmission"]
    },
    "Marketing": {
        "required_skills": ["market research", "digital marketing", "social media management", "branding", "advertising", "SEO", "content marketing", "data analysis", "email marketing"],
        "matching_keywords": ["advertisement", "marketing", "promotion", "branding", "social media", "digital marketing", "SEO", "market research"]
    },
    "Administration & Office Support": {
        "required_skills": ["data entry", "scheduling", "customer service", "office management", "email communication", "file management", "event coordination", "office software"],
        "matching_keywords": ["office", "administration", "data entry", "scheduling", "email", "reception", "management", "customer support"]
    },
    "Accounting": {
        "required_skills": ["financial reporting", "budgeting", "taxation", "accounting software", "bookkeeping", "auditing", "financial analysis", "regulatory compliance"],
        "matching_keywords": ["accounting", "finance", "bookkeeping", "audit", "tax", "budgeting", "financial reports"]
    },
    "Sport & Recreation": {
        "required_skills": ["athletic training", "fitness", "coaching", "sports management", "team leadership", "event planning", "sports marketing", "nutrition"],
        "matching_keywords": ["sports", "athletics", "fitness", "recreation", "coaching", "team leadership", "sports events"]
    },
    "Advertising, Arts & Media": {
        "required_skills": ["creativity", "graphic design", "media planning", "public relations", "advertising", "content creation", "video production", "marketing"],
        "matching_keywords": ["advertising", "arts", "media", "graphic design", "content creation", "public relations", "video production"]
    }
}

# ------------------- Experta Setup -------------------
class CareerExpertSystem(KnowledgeEngine):
    @DefFacts()
    def _initial_action(self):
        # Setup initial facts
        yield Fact(action="find_career_field")
    
    @Rule(Fact(action='find_career_field'), salience=100)
    def extract_skills(self):
        # Extract skills from the input text (for simplicity, we will match keywords from the user input)
        text_input = "I am good at selling products and negotiating with clients."
        self.declare(Fact(text_input=text_input))
    
    @Rule(Fact(text_input=MATCH.text_input), salience=90)
    def match_keywords_to_fields(self, text_input):
        # Match input text to career fields
        matched_fields = []
        for field, data in career_fields.items():
            for keyword in data["matching_keywords"]:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_input.lower()):
                    matched_fields.append(field)
                    break
        if matched_fields:
            for field in matched_fields:
                self.declare(Fact(matched_field=field))
    
    @Rule(Fact(matched_field=MATCH.field))
    def output_field(self, field):
        print(f"✅ Suitable career field: {field}")
        print(f"Required skills for {field}: {', '.join(career_fields[field]['required_skills'])}")
    

# -------------------- Run the Expert System -------------------
engine = CareerExpertSystem()
engine.reset()  # Prepare the system
engine.run()    # Run the inference engine
