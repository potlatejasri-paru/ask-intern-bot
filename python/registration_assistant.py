"""
AI Registration Assistant — NLTK reference implementation
Task AI-SS-001 · Student Code DAS008940

Mirrors the logic that powers the web chatbot in this repo:
    preprocess -> intent classification -> entity extraction -> validation
    -> dialog state machine -> JSON storage (no database).

Setup:
    pip install nltk
    python registration_assistant.py
"""

import json
import os
import re
import uuid
from datetime import datetime

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:  # offline fallback
        pass

DATA_FILE = "registrations.json"

INTENTS = {
    "greeting": {
        "patterns": ["hi", "hello", "hey", "good morning", "good evening", "namaste"],
        "response": "Hello! Welcome to the Free Online AI & Data Science Internship registration. "
                    "Type 'register' to begin, or ask me anything.",
    },
    "register": {
        "patterns": ["register", "registration", "apply", "sign up", "join", "enroll", "start"],
        "response": "Great! Let's get you registered.",
    },
    "help": {
        "patterns": ["help", "support", "assist", "guide", "options"],
        "response": "I can help with registration, duration, fees, certificates and required skills.",
    },
    "faq_duration": {
        "patterns": ["duration", "how long", "weeks", "timeline", "days"],
        "response": "The task runs for 7 days, roughly 3 hours per day.",
    },
    "faq_fee": {
        "patterns": ["fee", "cost", "price", "free", "payment", "charge"],
        "response": "The internship is completely free.",
    },
    "faq_certificate": {
        "patterns": ["certificate", "certification", "letter", "proof"],
        "response": "Yes — a completion certificate is issued after your submission is verified.",
    },
    "faq_skills": {
        "patterns": ["skill", "requirement", "prerequisite", "eligibility", "python"],
        "response": "Basic Python is enough; you'll learn NLP, intents and entity extraction here.",
    },
    "status": {"patterns": ["status", "progress", "summary", "my data"], "response": ""},
    "restart": {"patterns": ["restart", "reset", "start over", "clear"], "response": "Cleared. Type 'register' to start again."},
    "thank_you": {"patterns": ["thank", "thanks", "appreciate"], "response": "You're welcome!"},
    "goodbye": {"patterns": ["bye", "goodbye", "exit", "quit"], "response": "Goodbye and good luck!"},
}

SLOTS = ["name", "email", "phone", "field", "experience"]
PROMPTS = {
    "name": "What is your full name?",
    "email": "What's your email address?",
    "phone": "What's your 10-digit phone number?",
    "field": "Which field of study are you in?",
    "experience": "Describe your programming experience (beginner / intermediate / advanced).",
}

POSITIVE = {"good", "great", "awesome", "excited", "happy", "love", "nice", "thanks", "perfect", "yes"}
NEGATIVE = {"bad", "confused", "stuck", "angry", "hate", "worst", "difficult", "hard", "frustrated", "problem"}


class RegistrationAssistant:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            self.stop_words = {"a", "an", "the", "is", "am", "are", "i", "my", "to", "for", "of"}
        self.data = {}
        self.collecting = False

    # ------------------------------------------------------------ NLP layer
    def preprocess(self, text):
        text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        try:
            tokens = nltk.word_tokenize(text)
        except LookupError:
            tokens = text.split()
        return [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words]

    def classify_intent(self, text):
        """Bag-of-words overlap scoring, normalised by pattern count."""
        lower, tokens, best, best_score = text.lower(), self.preprocess(text), "unknown", 0.0
        for name, cfg in INTENTS.items():
            score = 0.0
            for pattern in cfg["patterns"]:
                if " " in pattern:
                    score += 1 if pattern in lower else 0
                elif self.lemmatizer.lemmatize(pattern) in tokens or re.search(rf"\b{pattern}\b", lower):
                    score += 1
            score /= len(cfg["patterns"]) ** 0.5
            if score > best_score:
                best, best_score = name, score
        return (best, round(best_score, 2)) if best_score >= 0.3 else ("unknown", round(best_score, 2))

    @staticmethod
    def extract_entities(text):
        entities = {}
        m = re.search(r"(?:my name is|i am|i'm|this is)\s+([a-zA-Z][a-zA-Z.\s]{1,49})", text, re.I)
        if m:
            entities["name"] = m.group(1).strip().title()
        m = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        if m:
            entities["email"] = m.group().lower()
        m = re.search(r"(?:\+\d{1,3}[\s-]?)?\d{10}\b", text)
        if m:
            entities["phone"] = re.sub(r"[\s-]", "", m.group())
        return entities

    @staticmethod
    def sentiment(text):
        words = set(re.sub(r"[^a-z\s]", " ", text.lower()).split())
        pos, neg = len(words & POSITIVE), len(words & NEGATIVE)
        return "positive" if pos > neg else "negative" if neg > pos else "neutral"

    # ---------------------------------------------------------- validation
    @staticmethod
    def validate(slot, value):
        value = value.strip()
        if slot == "name":
            return None if re.fullmatch(r"[a-zA-Z][a-zA-Z.\s]{2,49}", value) and len(value.split()) >= 2 \
                else "Please enter your first and last name (letters only)."
        if slot == "email":
            return None if re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", value) \
                else "Invalid email. Use a format like you@example.com."
        if slot == "phone":
            return None if re.fullmatch(r"(\+\d{1,3})?\d{10}", re.sub(r"[\s-]", "", value)) \
                else "Please enter a valid 10-digit phone number."
        return None if len(value) >= 2 else "Please provide a bit more detail."

    # ------------------------------------------------------ dialog manager
    def next_slot(self):
        return next((s for s in SLOTS if s not in self.data), None)

    def respond(self, text):
        intent, confidence = self.classify_intent(text)
        entities = self.extract_entities(text)
        print(f"   [debug] intent={intent} conf={confidence} entities={entities} sentiment={self.sentiment(text)}")

        if intent == "restart":
            self.data, self.collecting = {}, False
            return INTENTS["restart"]["response"]
        if intent == "status":
            return json.dumps(self.data, indent=2) if self.data else "Nothing collected yet."
        if intent in ("help", "faq_duration", "faq_fee", "faq_certificate", "faq_skills", "thank_you", "goodbye"):
            tail = f"\nBack to it — {PROMPTS[self.next_slot()]}" if self.collecting and self.next_slot() else ""
            return INTENTS[intent]["response"] + tail

        if not self.collecting:
            if intent == "register" or entities:
                self.collecting = True
                self.data.update({k: v for k, v in entities.items() if not self.validate(k, v)})
                slot = self.next_slot()
                return f"{INTENTS['register']['response']} {PROMPTS[slot]}" if slot else self.confirm()
            if intent == "greeting":
                return INTENTS["greeting"]["response"]
            return "I didn't catch that. Type 'register' to begin or 'help' for options."

        slot = self.next_slot()
        candidate = entities.get(slot, text.strip())
        error = self.validate(slot, candidate)
        if error:
            return error
        self.data[slot] = candidate.lower() if slot == "email" else candidate.title()
        nxt = self.next_slot()
        return f"Saved. {PROMPTS[nxt]}" if nxt else self.confirm()

    # ------------------------------------------------------ persistence
    def confirm(self):
        record = dict(
            id=f"REG-{uuid.uuid4().hex[:5].upper()}",
            **self.data,
            registered_at=datetime.now().isoformat(timespec="seconds"),
            task_id="AI-SS-001",
            student_code="DAS008940",
        )
        registrations = []
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE) as fh:
                registrations = json.load(fh)
        registrations.append(record)
        with open(DATA_FILE, "w") as fh:
            json.dump(registrations, fh, indent=2)
        self.collecting = False
        return "Registration confirmed!\n" + json.dumps(record, indent=2)


if __name__ == "__main__":
    bot = RegistrationAssistant()
    print("Assistant:", INTENTS["greeting"]["response"])
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower().strip() in ("quit", "exit", "bye"):
            print("Assistant: Thank you for using the AI Registration Assistant. Goodbye!")
            break
        print("Assistant:", bot.respond(user_input))
