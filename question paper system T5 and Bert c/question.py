from flask import Flask, render_template, request, send_file, redirect, url_for,session,jsonify,send_from_directory
import os
import random
import re
import string
import nltk
nltk.download("wordnet")
nltk.download('punkt')
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import sent_tokenize
import pke
from flashtext import KeywordProcessor
##from similarity.normalized_levenshtein import NormalizedLevenshtein
from pywsd.lesk import adapted_lesk
from pywsd.similarity import max_similarity
from docx import Document
import requests
import sklearn
# to use it like 
from sklearn.metrics.pairwise import cosine_similarity
import webbrowser
import pikepdf
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph,Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sqlite3
from fpdf import FPDF
import torch
from reportlab.pdfgen import canvas
import textwrap
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
import numpy as np
import Levenshtein
from sentence_transformers import SentenceTransformer
import zipfile

import spacy
nlp = spacy.load("en_core_web_sm")
import en_core_web_sm
nlp = en_core_web_sm.load()


app = Flask(__name__)
app.secret_key = "123"


UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

DATABASE = "database.db"

conn = sqlite3.connect(DATABASE)
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS register (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        usermail TEXT UNIQUE,
        password TEXT
    )
""")
conn.commit()
conn.close()

T5_MODEL_NAME = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME)

BERT_MODEL_NAME = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template('home.html')

    
@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    return render_template('features.html')  

@app.route('/faqs')
def faqs():
    return render_template('faqs.html')  

@app.route('/user_register', methods=['GET', 'POST'])
def user_register():
    if request.method == 'POST':
        username = request.form['name']
        usermail = request.form['email']
        password = request.form['password']

        # Validate username (only letters allowed)
        if not re.match("^[A-Za-z]+$", username):
            return render_template('index.html', error="Username must only contain letters")

        # Validate email (must be a Gmail address)
        if not re.match("^[a-zA-Z0-9]+[a-zA-Z0-9._%+-]*@gmail\.com$", usermail):
            return render_template('home.html', error="Email must be a valid Gmail address")

        # Check if the email already exists in the database
        conn = sqlite3.connect(DATABASE)
        cur = conn.cursor()
        cur.execute("SELECT * FROM register WHERE usermail = ?", (usermail,))
        data = cur.fetchone()

        if data:
            # If email exists, show an alert message
            return render_template('register.html', alert_message="Email already exists")

        # Add user to the database if everything is valid
        cur.execute("INSERT INTO register (username, usermail, password) VALUES (?, ?, ?)", (username, usermail, password))
        conn.commit()

        # Return success message after registration
        return render_template('register.html', alert_message="Registered successfully")
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        usermail = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect(DATABASE)
        cur = conn.cursor()
        cur.execute("SELECT * FROM register WHERE usermail = ? AND password = ?", (usermail, password))
        data = cur.fetchone()
        if data:
            session['email'] = usermail
            return render_template("dashboard.html")
        else:
            return render_template("register.html",alert_message="Incorrect Email & Password")
    return render_template('register.html')



def getImportantWords(art):
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=art, language="en")
    pos = {"NOUN", "PROPN"}
    stops = list(string.punctuation)
    stops += ["-lrb-", "-rrb-", "-lcb-", "-rcb-", "-lsb-", "-rsb-"]
    stops += stopwords.words("english")
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting()
    result = []
    ex = extractor.get_n_best(n=25)
    for each in ex:
        result.append(each[0])
    return result

def splitTextToSents(art):
    s = [sent_tokenize(art)]
    s = [y for x in s for y in x]
    s = [sent.strip() for sent in s if len(sent) > 15]
    return s

def mapSents(impWords, sents):
    processor = KeywordProcessor()
    keySents = {}
    for word in impWords:
        keySents[word] = []
        processor.add_keyword(word)
    for sent in sents:
        found = processor.extract_keywords(sent)
        for each in found:
            keySents[each].append(sent)
    for key in keySents.keys():
        temp = keySents[key]
        temp = sorted(temp, key=len, reverse=True)
        keySents[key] = temp
    return keySents

def getWordSense(sent, word):
    word = word.lower()
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    synsets = wn.synsets(word, "n")
    if synsets:
        wup = max_similarity(sent, word, "wup", pos="n")
        adapted_lesk_output = adapted_lesk(sent, word, pos="n")
        lowest_index = min(synsets.index(wup), synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None

def getDistractors(syn, word):
    dists = []
    word = word.lower()
    actword = word
    if len(word.split()) > 0:
        word.replace(" ", "_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
        return dists
    for each in hypernym[0].hyponyms():
        name = each.lemmas()[0].name()
        if name == actword:
            continue
        name = name.replace("_", " ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in dists:
            dists.append(name)
    return dists

def getDistractors2(word):
    word = word.lower()
    actword = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    dists = []
    url = f"http://api.conceptnet.io/query?node=/c/en/{word}/n&rel=/r/PartOf&start=/c/en/{word}&limit=5"
    obj = requests.get(url).json()
    for edge in obj["edges"]:
        link = edge["end"]["term"]
        url2 = f"http://api.conceptnet.io/query?node={link}&rel=/r/PartOf&end={link}&limit=10"
        obj2 = requests.get(url2).json()
        for edge in obj2["edges"]:
            word2 = edge["start"]["label"]
            if word2 not in dists and actword.lower() not in word2.lower():
                dists.append(word2)
    return dists

#summary
def clean_text(text):
    """Removes unwanted artifacts and extra spaces."""
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'[^A-Za-z0-9.,!?\'"() ]', '', text)  # Keep only meaningful characters
    return text.strip()

def preserve_named_entities(text):
    """Ensures named entities remain intact in summarization."""
    doc = nlp(text)
    named_entities = {ent.text: ent.label_ for ent in doc.ents}

    for entity, label in named_entities.items():
        text = text.replace(entity, f"[{label}] {entity}")

    return text, named_entities

def restore_named_entities(summary, named_entities):
    """Restores named entities in the summary output."""
    for entity, label in named_entities.items():
        summary = summary.replace(f"[{label}] {entity}", entity)
    return summary

def chunk_text(text, max_tokens=512):
    """Splits long texts into manageable chunks for T5 processing."""
    sentences = sent_tokenize(text)  # Split into sentences
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(sentence_tokens)

        if current_length + sentence_length > max_tokens:
            chunks.append(" ".join(current_chunk))  # Save the chunk
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:  
        chunks.append(" ".join(current_chunk))  # Add the last chunk

    return chunks


def capitalize_summary(summary):
    sentences = sent_tokenize(summary)
    sentences = [s[0].upper() + s[1:] if len(s) > 1 else s.upper() for s in sentences]
    return " ".join(sentences)

def summarize_text(text, max_length=150):
    """Summarizes text using T5, handling long inputs with chunking."""
    text = clean_text(text)
    text, named_entities = preserve_named_entities(text)

    # Split into chunks if text is too long
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        input_text = "summarize: " + chunk
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
        summary_ids = model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
        chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(chunk_summary)

    summary = ' '.join(summaries)  # Changed from `final_summary` to `summary`
    summary = restore_named_entities(summary, named_entities)
    summary = capitalize_summary(summary)

    return summary  # Returning `summary` instead of `final_summary`
    


@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            # Summarize the uploaded text using T5
            summary = summarize_text(text)

            impWords = getImportantWords(text)
            sents = splitTextToSents(text)
            mappedSents = mapSents(impWords, sents)

            mappedDists = {}
            for each in mappedSents:
                wordsense = getWordSense(mappedSents[each][0], each)
                if wordsense:
                    dists = getDistractors(wordsense, each)
                    if len(dists) == 0:
                        dists = getDistractors2(each)
                    if len(dists) != 0:
                        mappedDists[each] = dists
                else:
                    dists = getDistractors2(each)
                    if len(dists) > 0:
                        mappedDists[each] = dists

            question_content = []
            answer_content = []
            styles = getSampleStyleSheet()

            iterator = 1
            answer_key = []
            for each in mappedDists:
                sent = mappedSents[each][0]
                p = re.compile(each, re.IGNORECASE)
                op = p.sub("________", sent)

                correct_answer = each.capitalize()
                options = [each.capitalize()] + mappedDists[each]
                options = options[:4]
                random.shuffle(options)

                # Store correct option label
                correct_label = chr(ord('a') + options.index(correct_answer))
                answer_key.append((iterator, f"{correct_label}) {correct_answer}"))

                question_text = f"Question {iterator}: {op}"
                question_content.append(Paragraph(question_text, styles["Normal"]))
                question_content.append(Spacer(1, 12))

                for i, option in enumerate(options):
                    option_text = f"{chr(ord('a') + i)}) {option}"
                    question_content.append(Paragraph(option_text, styles["Normal"]))
                
                question_content.append(Spacer(1, 20))
                iterator += 1

            # Add the summary to the PDF
            question_content.insert(0, Paragraph("Summary of the Document:", styles["Heading1"]))
            question_content.insert(1, Paragraph(summary, styles["Normal"]))
            question_content.insert(2,Spacer(1, 20))

            # Add answer key section
            answer_content.append(Paragraph("Answer Key:", styles["Heading1"]))
            answer_content.append(Spacer(1, 12))
            for q_num, ans in answer_key:
                answer_content.append(Paragraph(f"{q_num}) {ans}", styles["Normal"]))
                answer_content.append(Spacer(1, 6))

            # Create PDFs
            questions_pdf_path = os.path.join(UPLOAD_FOLDER, "mcq_questions.pdf")
            answers_pdf_path = os.path.join(UPLOAD_FOLDER, "mcq_answer_key.pdf")

            SimpleDocTemplate(questions_pdf_path, pagesize=letter).build(question_content)
            SimpleDocTemplate(answers_pdf_path, pagesize=letter).build(answer_content)

            # Zip both PDFs
            zip_path = os.path.join(UPLOAD_FOLDER, "mcq_bundle.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(questions_pdf_path, arcname="MCQ_Questions.pdf")
                zipf.write(answers_pdf_path, arcname="Answer_Key.pdf")

            return send_file(zip_path, as_attachment=True)

            '''pdf_filename = os.path.join(UPLOAD_FOLDER, "mcq_questions_with_summary.pdf")
            pdf = SimpleDocTemplate(pdf_filename, pagesize=letter)
            pdf.build(questions_list)

            return send_file(pdf_filename, as_attachment=True)'''

    return render_template("index.html")


'''def generate_questions(text):
    sentences = text.split(". ")  # Simple sentence split
    questions = []
    
    for i, sentence in enumerate(sentences):
        if len(sentence.split()) > 5:
            questions.append(f"Question {i+1}: {sentence}?")
    
    return questions'''

def generate_questions(text):
    sentences = sent_tokenize(text)  # Use NLTK for better sentence splitting
    questions = []

    question_starters = [
        "What", "How", "Why", "When", "Where", "Which", "Who", "Can", "Is", "Are", "Does"
    ]

    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if len(sentence.split()) > 5 and sentence[-1] != "?":
            # Turn the sentence into a question if it's not already
            if sentence.split()[0] not in question_starters:
                sentence += "?"
            questions.append(f"Question {i + 1}: {sentence}")

    return questions


@app.route("/question", methods=["GET"])
def question():
    return render_template("question.html")


@app.route("/generate_questions", methods=["POST"])
def generate_questions_pdf():
    if "file" not in request.files:
        return "No file uploaded", 400
    
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    questions = generate_questions(text)
    pdf_filename = os.path.join(UPLOAD_FOLDER, "generated_questions.pdf")
    pdf = SimpleDocTemplate(pdf_filename, pagesize=letter)
    content = []
    styles = getSampleStyleSheet()
    
    for q in questions:
        content.append(Paragraph(q, styles["Normal"]))
        content.append(Spacer(1, 20))
    
    pdf.build(content)
    return send_file(pdf_filename, as_attachment=True)

@app.route("/fillups", methods=["GET"])
def fillups():
    return render_template("fillups.html")

'''def generate_fill_in_the_blank(text):
    sentences = text.split(". ")  # Simple sentence splitting
    fill_in_blanks = []
    
    for sentence in sentences:
        words = sentence.split()
        if len(words) > 5:
            word_to_remove = random.choice(words)  # Pick a random word to replace
            blank_sentence = sentence.replace(word_to_remove, "_____")
            fill_in_blanks.append(blank_sentence)

    return fill_in_blanks'''

'''def generate_fill_in_the_blank(text):
    sentences = sent_tokenize(text)  # Better sentence splitting
    fill_in_blanks = []
    used_sentences = set()

    for sentence in sentences:
        words = sentence.split()
        if len(words) > 6:
            # Choose a non-trivial word (skip articles, prepositions, etc.)
            content_words = [word for word in words if len(word) > 3 and word.isalpha()]
            if not content_words:
                continue

            word_to_remove = random.choice(content_words)
            if sentence in used_sentences:
                continue

            blank_sentence = sentence.replace(word_to_remove, "_____")
            fill_in_blanks.append(blank_sentence.strip())
            used_sentences.add(sentence)

    return fill_in_blanks'''

from nltk.tokenize import sent_tokenize, word_tokenize
import random
stop_words = set(stopwords.words('english'))

def generate_fill_in_the_blank(text):
    sentences = sent_tokenize(text)  # Split into sentences
    fill_in_blanks = []
    answer_key = []
    used_sentences = set()

    for sentence in sentences:
        words = word_tokenize(sentence)
        if len(words) > 6:
            # Filter out short words and punctuation
            candidates = [w for w in words if w.isalpha() and len(w) > 3 and w.lower() not in stop_words]

            if not candidates:
                continue

            word_to_blank = random.choice(candidates)
            pattern = r'\b' + re.escape(word_to_blank) + r'\b'
            blanked_sentence = re.sub(pattern, "_____", sentence, count=1)

            if sentence not in used_sentences:
                fill_in_blanks.append(blanked_sentence.strip())
                answer_key.append((len(fill_in_blanks), word_to_blank))  # (question number, answer)
                used_sentences.add(sentence)

    return fill_in_blanks, answer_key


@app.route("/generate_fill_in_blanks", methods=["POST"])
def generate_fill_in_blanks_pdf():
    if "file" not in request.files:
        return "No file uploaded", 400
    
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Get fillups and answers
    fill_in_blanks, answer_key = generate_fill_in_the_blank(text)
    
    '''fill_in_blanks = generate_fill_in_the_blank(text)
    pdf_filename = os.path.join(UPLOAD_FOLDER, "fill_in_the_blank_questions.pdf")
    pdf = SimpleDocTemplate(pdf_filename, pagesize=letter)
    content = []
    styles = getSampleStyleSheet()
    
    for q in fill_in_blanks:
        content.append(Paragraph(q, styles["Normal"]))
        content.append(Spacer(1, 20))'''
    
    # Create Question PDF
    question_pdf_path = os.path.join(UPLOAD_FOLDER, "fill_in_the_blank_questions.pdf")
    question_doc = SimpleDocTemplate(question_pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    question_content = []
    
    for i, q in enumerate(fill_in_blanks, 1):  # No slicing, no limit
        question_content.append(Paragraph(f"{i}) {q}", styles["Normal"]))
        question_content.append(Spacer(1, 12))

    question_doc.build(question_content)

    # Create Answer Key PDF
    answer_pdf_path = os.path.join(UPLOAD_FOLDER, "fill_in_the_blank_answer_key.pdf")
    answer_doc = SimpleDocTemplate(answer_pdf_path, pagesize=letter)
    answer_content = []

    answer_content.append(Paragraph("Answer Key", styles["Heading1"]))
    answer_content.append(Spacer(1, 12))

    for q_num, ans in answer_key:
        answer_content.append(Paragraph(f"{q_num}) {ans}", styles["Normal"]))
        answer_content.append(Spacer(1, 6))

    answer_doc.build(answer_content)

    # Optionally: Return both PDFs as a downloadable zip
    
    zip_path = os.path.join(UPLOAD_FOLDER, "fillups_bundle.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(question_pdf_path, arcname="Questions.pdf")
        zipf.write(answer_pdf_path, arcname="Answer_Key.pdf")

    return send_file(zip_path, as_attachment=True)
    
    '''pdf.build(content)
    return send_file(pdf_filename, as_attachment=True)'''



@app.route("/synonyms", methods=["GET"])
def synonyms():
    return render_template("synonyms.html")


PREDEFINED_SYNONYMS = {
    "Difference": [
        "Distinction", "Discrepancy", "Variation", "Contrast", "Divergence",
        "Gap", "Dissimilarity", "Alteration", "Variation", "Distinction",
        "Opposition", "Diversity", "Imbalance", "Conflict", "Change",
        "Deviation", "Polarity", "Disparity", "Split", "Contradiction"
    ],
    "Important": [
        "Significant", "Essential", "Crucial", "Vital", "Necessary",
        "Major", "Relevant", "Noteworthy", "Fundamental", "Substantial",
        "Key", "Critical", "Influential", "Valuable", "Meaningful",
        "Momentous", "Pivotal", "Serious", "Central", "Imperative"
    ],
    "New": [
        "Recent", "Fresh", "Modern", "Latest", "Novel",
        "Innovative", "Original", "Contemporary", "Unfamiliar", "Advanced",
        "Progressive", "Trendy", "Current", "Up-to-date", "Latest",
        "Unique", "Pioneering", "Next-gen", "Experimental", "Groundbreaking"
    ],
    "Thinking": [
        "Contemplation", "Reflection", "Consideration", "Pondering", "Meditation",
        "Reasoning", "Analysis", "Thought", "Evaluation", "Speculation",
        "Cognition", "Deliberation", "Brainstorming", "Insight", "Interpretation",
        "Judgment", "Understanding", "Introspection", "Perception", "Awareness"
    ],
    "Story": [
        "Tale", "Narrative", "Account", "Anecdote", "Chronicle",
        "Fable", "Legend", "Myth", "Saga", "Plot",
        "Report", "Biography", "History", "Memoir", "Parable",
        "Confession", "Yarn", "Fiction", "Drama", "Allegory"
    ],
    "Sad": [
        "Unhappy", "Depressed", "Sorrowful", "Mournful", "Melancholy",
        "Gloomy", "Downcast", "Despondent", "Dismal", "Heartbroken",
        "Dejected", "Forlorn", "Grief-stricken", "Wretched", "Blue",
        "Woeful", "Low-spirited", "Disheartened", "Doleful", "Morose"
    ],
    "Strong": [
        "Powerful", "Mighty", "Robust", "Resilient", "Hardy",
        "Tough", "Sturdy", "Forceful", "Dominant", "Stalwart",
        "Unyielding", "Vigorous", "Potent", "Staunch", "Solid",
        "Steadfast", "Durable", "Intense", "Firm", "Indomitable"
    ],
    "Fast": [
        "Quick", "Rapid", "Speedy", "Swift", "Brisk",
        "Hasty", "Expeditious", "Nimble", "Prompt", "Fleet",
        "Accelerated", "Hurried", "Sprightly", "Breakneck", "Snappy",
        "High-speed", "Express", "Immediate", "Lively", "Turbocharged"
    ],
    "Smart": [
        "Intelligent", "Clever", "Bright", "Brilliant", "Quick-witted",
        "Sharp", "Witty", "Wise", "Ingenious", "Astute",
        "Savvy", "Perceptive", "Erudite", "Brainy", "Shrewd",
        "Insightful", "Resourceful", "Knowledgeable", "Gifted", "Rational"
    ],"Algorithm": [
        "Procedure", "Method", "Routine", "Process", "Formula",
        "Blueprint", "Framework", "Logic", "Model", "Workflow",
        "Pattern", "Sequence", "Program", "Protocol", "Computation",
        "Function", "Rule set", "Technique", "Strategy", "System"
    ],
    "Database": [
        "Data store", "Repository", "Archive", "Ledger", "Storage",
        "Index", "Catalog", "Register", "Table", "Record system",
        "Data warehouse", "Information system", "Data structure", "File system", "Schema",
        "Server", "Dataset", "Memory bank", "Knowledge base", "Data mart"
    ],
    "Encryption": [
        "Cryptography", "Encoding", "Ciphering", "Obfuscation", "Scrambling",
        "Hashing", "Tokenization", "Securing", "Masking", "Anonymization",
        "Digital sealing", "Data protection", "Code transformation", "Secrecy enforcement", "Authentication",
        "Public-key cryptography", "Private-key cryptography", "Symmetric encryption", "Asymmetric encryption", "Encoding process"
    ],
    "Machine Learning": [
        "AI training", "Pattern recognition", "Predictive modeling", "Data mining", "Neural networks",
        "Deep learning", "Supervised learning", "Unsupervised learning", "Reinforcement learning", "Algorithmic training",
        "Classification", "Regression", "Feature engineering", "Data analysis", "Statistical modeling",
        "Big data processing", "Cognitive computing", "Pattern detection", "Artificial intelligence", "Self-learning systems"
    ],
    "Programming": [
        "Coding", "Development", "Software engineering", "Scripting", "Application development",
        "System design", "Software design", "Automation", "Computing", "Tech development",
        "Software architecture", "Debugging", "Implementation", "Algorithm design", "Software writing",
        "Code optimization", "Technical writing", "Backend development", "Frontend development", "Code generation"
    ],
    "Cybersecurity": [
        "Information security", "Data protection", "Threat management", "Network security", "Risk mitigation",
        "Vulnerability assessment", "Encryption", "Firewall protection", "Access control", "Secure coding",
        "Intrusion detection", "Penetration testing", "Identity management", "Security compliance", "Data governance",
        "Forensics", "Malware defense", "Zero-trust security", "Cyber threat intelligence", "Authentication mechanisms"
    ],
    "Networking": [
        "Communication", "Packet transmission", "Protocol management", "Data routing", "LAN",
        "WAN", "Wireless networking", "Network topology", "Internet architecture", "Server-client model",
        "Cloud networking", "Subnets", "Network administration", "IP addressing", "Firewall management",
        "Switching", "Routing", "Network security", "VPN", "TCP/IP"
    ],
    "Artificial Intelligence": [
        "Machine intelligence", "Neural networks", "Cognitive computing", "Deep learning", "Expert systems",
        "Automation", "Predictive analysis", "Data science", "Smart systems", "Self-learning",
        "Intelligent automation", "Algorithmic reasoning", "AI models", "Automated decision-making", "Chatbots",
        "Computer vision", "Natural language processing", "Reinforcement learning", "AI-driven solutions", "AI applications"
    ]
}
@app.route("/get_synonyms", methods=["POST"])
def get_synonyms():
    data = request.get_json()
    word = data.get("word", "").strip().lower()

    if not word:
        return jsonify({"synonyms": []})

    # If word is one of the predefined ones, return directly
    if word in {k.lower(): v for k, v in PREDEFINED_SYNONYMS.items()}:
        return jsonify({"synonyms": PREDEFINED_SYNONYMS[word.capitalize()]})

    # For other words, generate synonyms using WordNet
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))

    synonyms_list = list(synonyms)

    # Ensure 20 suggestions (repeat if needed)
    while len(synonyms_list) < 20 and len(synonyms_list) > 0:
        synonyms_list.append(synonyms_list[len(synonyms_list) % len(synonyms_list)])

    return jsonify({"synonyms": synonyms_list[:20]})

@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            # Read file content
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            # Generate summary using T5 model
            #input_text = "summarize: " + text
            #input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            #summary_ids = model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
            #summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Generate summary
            summary = summarize_text(text)

            # Create PDF file
            pdf_filepath = os.path.join(UPLOAD_FOLDER, 'summary_' + file.filename.replace('.txt', '.pdf'))
            c = canvas.Canvas(pdf_filepath, pagesize=letter)
            width, height = letter

            c.setFont("Helvetica", 12)
            c.drawString(100, height - 50, "Summary:")

            y_position = height - 70
            line_height = 14

            # Wrap long lines to fit within PDF width
            wrapped_summary = textwrap.wrap(summary, width=80)  # Adjust width as needed

            for line in wrapped_summary:
                if y_position < 40:  # Create new page if running out of space
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y_position = height - 50
                c.drawString(50, y_position, line)
                y_position -= line_height

            c.save()

            # Return summarized PDF file for download
            return send_file(pdf_filepath, as_attachment=True)

    return render_template('summarize.html')

@app.route("/plagrasim", methods=["GET","POST"])
def plagrasim():
    return render_template("plagrasim.html")


'''def get_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs, output_hidden_states=True)
        outputs = bert_model(**inputs)
        
    return outputs.last_hidden_state.mean(dim=1).numpy()'''
 

def get_embedding(text):
    return similarity_model.encode(text, normalize_embeddings=True).reshape(1, -1)


@app.route('/similarity', methods=['GET', 'POST'])
def similarity():
    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1 and file2:
            filepath1 = os.path.join(UPLOAD_FOLDER, file1.filename)
            filepath2 = os.path.join(UPLOAD_FOLDER, file2.filename)
            file1.save(filepath1)
            file2.save(filepath2)

            with open(filepath1, 'r', encoding='utf-8') as f:
                text1 = f.read()
            with open(filepath2, 'r', encoding='utf-8') as f:
                text2 = f.read()

            # Generate embeddings using BERT
            emb1 = get_embedding(text1)
            emb2 = get_embedding(text2)
             
           

            # Calculate similarity score
            similarity_score = cosine_similarity(emb1, emb2)[0][0] * 100
            similarity_score = round(similarity_score, 2)

            # Calculate text difference
            difference_score = calculate_text_difference(text1, text2)

            # Generate similarity graph
            plt.figure(figsize=(6, 4))
            labels = ['Similarity', 'Difference']
            values = [similarity_score, difference_score]
            plt.bar(labels, values, color=['#4CAF50', '#FF6347'])
            plt.title('Similarity and Difference Score')
            plt.ylim(0, 100)

            graph_filename = 'similarity_score.png'
            graph_path = os.path.join(UPLOAD_FOLDER, graph_filename)
            plt.savefig(graph_path)
            plt.close()

            # Redirect to result page with score and graph
            return redirect(url_for('similarity_result', score=similarity_score, graph=graph_filename))

    return render_template('plagrasim.html')

@app.route('/similarity_result')
def similarity_result():
    score = request.args.get('score', type=float)
    graph = request.args.get('graph', default=None)

    return render_template('similarity_result.html', score=score, graph=f"/static/uploads/{graph}")

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def calculate_text_difference(text1, text2):
    distance = Levenshtein.distance(text1, text2)
    max_length = max(len(text1), len(text2))
    difference_percentage = (distance / max_length) * 100
    return round(difference_percentage, 2)

if __name__ == "__main__":
    app.run(debug=False,port=1050)
