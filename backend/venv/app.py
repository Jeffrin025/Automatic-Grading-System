# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename
# import os
# import pytesseract
# from PIL import Image
# import re
# import torch
# from transformers import BertTokenizer, BertModel
# import warnings

# # Suppress warnings
# warnings.filterwarnings("ignore")

# app = Flask(__name__)

# # Configure upload folder
# UPLOAD_FOLDER = 'uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Initialize device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # Define your EXACT custom model architecture based on the error messages
# class CustomBertScorer(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
        
#         # Adjusted to match your actual model dimensions
#         self.total_mark_embed = torch.nn.Linear(1, 32)  # Changed from 768 to 32
#         self.regressor = torch.nn.Linear(768 + 32, 1)  # 768 (BERT) + 32 (mark) = 800
        
#     def forward(self, input_ids, attention_mask, max_score):
#         # Get BERT outputs
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        
#         # Process max_score (now with correct dimension)
#         mark_embedded = self.total_mark_embed(max_score.unsqueeze(-1))
        
#         # Combine features
#         combined = torch.cat([pooled_output, mark_embedded], dim=1)
#         return self.regressor(combined)

# # Load your custom model
# try:
#     model = CustomBertScorer().to(device)
#     model.load_state_dict(torch.load('bert_regressor.pth', map_location=device))
#     model.eval()
#     print("Model loaded successfully with correct architecture")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)

# def extract_text_from_image(image_path):
#     """Extract text from image using OCR"""
#     try:
#         img = Image.open(image_path)
#         text = pytesseract.image_to_string(img)
#         return text
#     except Exception as e:
#         print(f"Error in OCR: {e}")
#         return ""

# def parse_qa_pairs(text):
#     """Parse questions and answers with marks"""
#     questions = re.split(r'(?i)q\d+\.', text)
#     questions = [q.strip() for q in questions if q.strip()]
    
#     qa_pairs = []
#     for q in questions:
#         mark_match = re.search(r'\[(\d+)\]$', q)
#         max_score = int(mark_match.group(1)) if mark_match else 1
#         answer = re.sub(r'\[\d+\]$', '', q).strip()
#         qa_pairs.append({'answer': answer, 'max_score': max_score})
    
#     return qa_pairs

# def predict_mark(answer_key, student_answer, max_score):
#     """Predict score using your custom model"""
#     inputs = tokenizer(
#         answer_key,
#         student_answer,
#         max_length=512,
#         padding='max_length',
#         truncation=True,
#         return_tensors="pt"
#     ).to(device)
    
#     max_score_tensor = torch.tensor([[max_score]], dtype=torch.float).to(device)
    
#     with torch.no_grad():
#         predicted_score = model(
#             input_ids=inputs['input_ids'],
#             attention_mask=inputs['attention_mask'],
#             max_score=max_score_tensor
#         ).item()
    
#     return min(max(0, predicted_score), max_score)
# @app.route('/upload', methods=['POST'])
# def upload_files():
#     print("📥 Received POST request to /upload")
    
#     # Check if files are part of the request
#     if 'answerKey' not in request.files:
#         print("❌ 'answerKey' not found in request.files")
#     if 'studentAnswers' not in request.files:
#         print("❌ 'studentAnswers' not found in request.files")
    
#     if 'answerKey' not in request.files or 'studentAnswers' not in request.files:
#         return jsonify({'error': 'Missing files'}), 400

#     # Get file lists
#     answer_key_files = request.files.getlist('answerKey')
#     student_answer_files = request.files.getlist('studentAnswers')

#     print(f"📄 Number of answerKey files: {len(answer_key_files)}")
#     print(f"📄 Number of studentAnswers files: {len(student_answer_files)}")

#     if not answer_key_files or not student_answer_files:
#         print("❌ No files selected")
#         return jsonify({'error': 'No files selected'}), 400

#     try:
#         # Save and extract answer key
#         answer_key_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(answer_key_files[0].filename))
#         answer_key_files[0].save(answer_key_path)
#         print(f"✅ Saved answer key to: {answer_key_path}")
        
#         answer_key_text = extract_text_from_image(answer_key_path)
#         print(f"🧾 Extracted answer key text:\n{answer_key_text}")
        
#         answer_key_data = parse_qa_pairs(answer_key_text)
#         print(f"📊 Parsed {len(answer_key_data)} questions from answer key")

#         # Save and extract student answers
#         student_answer_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(student_answer_files[0].filename))
#         student_answer_files[0].save(student_answer_path)
#         print(f"✅ Saved student answers to: {student_answer_path}")

#         student_answer_text = extract_text_from_image(student_answer_path)
#         print(f"🧾 Extracted student answer text:\n{student_answer_text}")
        
#         student_answer_data = parse_qa_pairs(student_answer_text)
#         print(f"📊 Parsed {len(student_answer_data)} questions from student answers")

#         # Validate question count
#         if len(answer_key_data) != len(student_answer_data):
#             print("❌ Question count mismatch")
#             return jsonify({
#                 'error': f'Question count mismatch. Key: {len(answer_key_data)}, Student: {len(student_answer_data)}'
#             }), 400

#         # Scoring
#         detailed_scores = {}
#         total_score = 0
#         total_marks = 0

#         for i in range(len(answer_key_data)):
#             q_num = i + 1
#             max_score = answer_key_data[i]['max_score']
#             key_answer = answer_key_data[i]['answer']
#             student_answer = student_answer_data[i]['answer']

#             print(f"🔍 Scoring Q{q_num} - Max Score: {max_score}")
#             predicted_score = predict_mark(key_answer, student_answer, max_score)
#             print(f"✅ Predicted score: {predicted_score}")

#             detailed_scores[f"Q{q_num}"] = {
#                 'score': round(predicted_score, 2),
#                 'max_score': max_score,
#                 'key_answer': key_answer,
#                 'student_answer': student_answer
#             }

#             total_score += predicted_score
#             total_marks += max_score

#         # Final response
#         response = {
#             'total_score': round(total_score, 2),
#             'total_marks': total_marks,
#             'percentage': round((total_score / total_marks) * 100, 2) if total_marks > 0 else 0,
#             'detailed_scores': detailed_scores
#         }

#         # Clean up files
#         if os.path.exists(answer_key_path):
#             os.remove(answer_key_path)
#         if os.path.exists(student_answer_path):
#             os.remove(student_answer_path)

#         print("✅ Successfully processed and scored")
#         return jsonify(response)

#     except Exception as e:
#         print(f"❌ Error processing files: {e}")
#         return jsonify({'error': str(e)}), 500


# if __name__ == '__main__':
#     app.run(debug=True)







# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename
# import os
# import re
# import torch
# import warnings
# from PIL import Image
# import pytesseract

# from transformers import BertTokenizer, BertModel, TrOCRProcessor, VisionEncoderDecoderModel

# # Suppress warnings
# warnings.filterwarnings("ignore")

# app = Flask(__name__)

# # Configure upload folder
# UPLOAD_FOLDER = 'uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Initialize device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # Load TrOCR model and processor
# print("🔁 Loading TrOCR processor and model...")
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)
# trocr_model.eval()
# print("✅ TrOCR loaded successfully")

# # Define your EXACT custom model architecture
# class CustomBertScorer(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.total_mark_embed = torch.nn.Linear(1, 32)
#         self.regressor = torch.nn.Linear(768 + 32, 1)

#     def forward(self, input_ids, attention_mask, max_score):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.last_hidden_state[:, 0, :]
#         mark_embedded = self.total_mark_embed(max_score.unsqueeze(-1))
#         combined = torch.cat([pooled_output, mark_embedded], dim=1)
#         return self.regressor(combined)

# # Load your custom model
# try:
#     model = CustomBertScorer().to(device)
#     model.load_state_dict(torch.load('bert_regressor.pth', map_location=device))
#     model.eval()
#     print("✅ Model loaded successfully with correct architecture")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     exit(1)

# def extract_text_from_image(image_path):
#     print(f"🔍 Performing OCR on image: {image_path}")
#     try:
#         image = Image.open(image_path).convert("RGB")
#         print(f"📐 Image size: {image.size}, mode: {image.mode}")

#         # Optional pre-processing
#         image = image.resize((int(image.width * 1.5), int(image.height * 1.5)))
#         image = image.convert('L')

#         # Run through TrOCR
#         pixel_values = processor(image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
#         generated_ids = trocr_model.generate(pixel_values)  # ✅ Correct model used here
#         generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

#         if generated_text.strip() == "" or generated_text.strip() == "0 0":
#             print("⚠️ TrOCR result is empty or nonsense, using pytesseract as fallback")
#             fallback_text = pytesseract.image_to_string(image)
#             print(f"🧾 OCR Output (Tesseract Fallback):\n{fallback_text}")
#             return fallback_text

#         print(f"🧾 OCR Output (TrOCR):\n{generated_text}")
#         return generated_text

#     except Exception as e:
#         print(f"❌ Error during OCR: {e}")
#         return ""

# def parse_qa_pairs(text):
#     """Parse questions and answers with marks"""
#     questions = re.split(r'(?i)q\d+\.', text)
#     questions = [q.strip() for q in questions if q.strip()]

#     qa_pairs = []
#     for q in questions:
#         mark_match = re.search(r'\[(\d+)\]$', q)
#         max_score = int(mark_match.group(1)) if mark_match else 1
#         answer = re.sub(r'\[\d+\]$', '', q).strip()
#         qa_pairs.append({'answer': answer, 'max_score': max_score})

#     return qa_pairs

# def predict_mark(answer_key, student_answer, max_score):
#     """Predict score using your custom model"""
#     inputs = tokenizer(
#         answer_key,
#         student_answer,
#         max_length=512,
#         padding='max_length',
#         truncation=True,
#         return_tensors="pt"
#     ).to(device)

#     max_score_tensor = torch.tensor([[max_score]], dtype=torch.float).to(device)

#     with torch.no_grad():
#         predicted_score = model(
#             input_ids=inputs['input_ids'],
#             attention_mask=inputs['attention_mask'],
#             max_score=max_score_tensor
#         ).item()

#     return min(max(0, predicted_score), max_score)

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     print("📥 Received POST request to /upload")

#     if 'answerKey' not in request.files or 'studentAnswers' not in request.files:
#         print("❌ Missing files")
#         return jsonify({'error': 'Missing files'}), 400

#     answer_key_files = request.files.getlist('answerKey')
#     student_answer_files = request.files.getlist('studentAnswers')

#     print(f"📄 Number of answerKey files: {len(answer_key_files)}")
#     print(f"📄 Number of studentAnswers files: {len(student_answer_files)}")

#     if not answer_key_files or not student_answer_files:
#         print("❌ No files selected")
#         return jsonify({'error': 'No files selected'}), 400

#     try:
#         # Process answer key
#         answer_key_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(answer_key_files[0].filename))
#         answer_key_files[0].save(answer_key_path)
#         print(f"✅ Saved answer key to: {answer_key_path}")

#         answer_key_text = extract_text_from_image(answer_key_path)
#         answer_key_data = parse_qa_pairs(answer_key_text)
#         print(f"📊 Parsed {len(answer_key_data)} questions from answer key")

#         # Process student answers
#         student_answer_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(student_answer_files[0].filename))
#         student_answer_files[0].save(student_answer_path)
#         print(f"✅ Saved student answers to: {student_answer_path}")

#         student_answer_text = extract_text_from_image(student_answer_path)
#         student_answer_data = parse_qa_pairs(student_answer_text)
#         print(f"📊 Parsed {len(student_answer_data)} questions from student answers")

#         # Validate matching Q count
#         if len(answer_key_data) != len(student_answer_data):
#             print("❌ Question count mismatch")
#             return jsonify({
#                 'error': f'Question count mismatch. Key: {len(answer_key_data)}, Student: {len(student_answer_data)}'
#             }), 400

#         # Scoring
#         detailed_scores = {}
#         total_score = 0
#         total_marks = 0

#         for i in range(len(answer_key_data)):
#             q_num = i + 1
#             max_score = answer_key_data[i]['max_score']
#             key_answer = answer_key_data[i]['answer']
#             student_answer = student_answer_data[i]['answer']

#             print(f"🔍 Scoring Q{q_num} - Max Score: {max_score}")
#             predicted_score = predict_mark(key_answer, student_answer, max_score)
#             print(f"✅ Predicted score: {predicted_score}")

#             detailed_scores[f"Q{q_num}"] = {
#                 'score': round(predicted_score, 2),
#                 'max_score': max_score,
#                 'key_answer': key_answer,
#                 'student_answer': student_answer
#             }

#             total_score += predicted_score
#             total_marks += max_score

#         response = {
#             'total_score': round(total_score, 2),
#             'total_marks': total_marks,
#             'percentage': round((total_score / total_marks) * 100, 2) if total_marks > 0 else 0,
#             'detailed_scores': detailed_scores
#         }

#         # Cleanup
#         if os.path.exists(answer_key_path):
#             os.remove(answer_key_path)
#         if os.path.exists(student_answer_path):
#             os.remove(student_answer_path)

#         print("✅ Successfully processed and scored")
#         return jsonify(response)

#     except Exception as e:
#         print(f"❌ Error processing files: {e}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



# The diqestive system breaks down food into nutrients that the body can
# absorb and use for enerqy and qrowth.

# The diqestive System helps break down food into Small parts So the body cAn
# absorb nutrients.


from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import warnings
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import google.generativeai as genai

# Suppress warnings
warnings.filterwarnings("ignore")

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)


app = Flask(__name__)
CORS(app)

genai.configure(api_key="AIzaSyAfseh-7FZMhEQ5PDzMM5aBVmfM-FQegd4")

# Give Gemini model a unique name
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Function that uses Gemini
def get_feedback_from_gemini(answer):
    prompt = f"""
You are an AI examiner. A student answered a subjective question.
Below is the answer key and the student's answer.

Answer Key and student answer : 
{answer}

Please provide:

1. Constructive feedback for the student when given the answer key and student answer .
2. Specific areas they should improve.
3. Do not repeat the answer key.
4. Be helpful, and assume the student is in school or college.

Format your feedback in 3-5 lines.
"""
    try:
        response = gemini_model.generate_content(prompt)
        feedback = response.text.strip()
        print("📘 Gemini Feedback:\n" + feedback + "\n" + "-" * 50)
        return feedback
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "Error generating feedback."

    
def rectify_answer_key_ocr(ocr_text):
    prompt = f"""
You are an AI text corrector. The following content was extracted using OCR from an answer key. It contains multiple subjective answers labeled by question (e.g., q1., q2., etc.), but it may include spelling mistakes, garbled text, and formatting issues.

Please do the following:
1. Correct all grammar, spelling, and OCR errors.
2. Keep each question labeled as lowercase like this: q1., q2., etc.
3. Ensure each answer ends with a mark in curly brackets like this: {2}, {5} .only 2 and 5 marks .
4. Do not fabricate or add any new information. Only correct what's legible and logical.
5. If an answer seems missing or too corrupted fix it .
6.If qns are jumbled like frist q10 then q1 etc.. order it correctly q1 then q2 then q3 like that .
7. The final mark at the end of the last question is important. Keep it exactly as it appears — do not remove or modify it.
8. Only output the cleaned, formatted text. Do not explain anything.

OCR Text:
{ocr_text}

Corrected and Structured Answer Key:
"""
    try:
        response = gemini_model.generate_content(prompt)
        cleaned_key = response.text.strip()
        print("✅ Cleaned Answer Key:\n" + cleaned_key + "\n" + "-" * 50)
        return cleaned_key
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "Error processing the OCR text."
    
def rectify_student_ocr(ocr_text):
    prompt = f"""
You are an AI text corrector. The following content was extracted using OCR from an answer key. It contains multiple subjective answers labeled by question (e.g., q1., q2., etc.), but it may include spelling mistakes, garbled text, and formatting issues.

Please do the following:
1. Correct all grammar, spelling, and OCR errors.
2. Keep each question labeled as lowercase like this: q1., q2., etc.
3. Do not fabricate or add any new information. Only correct what's legible and logical.
4. If an answer seems missing or too corrupted fix it .
5.If qns are jumbled like frist q10 then q1 etc.. order it correctly q1 then q2 then q3 like that .
6. Only output the cleaned, formatted text. Do not explain anything.

OCR Text:
{ocr_text}

Corrected and Structured Answer Key:
"""
    try:
        response = gemini_model.generate_content(prompt)
        cleaned_key = response.text.strip()
        print("✅ Cleaned Answer Key:\n" + cleaned_key + "\n" + "-" * 50)
        return cleaned_key
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "Error processing the OCR text."


# Upload folder config
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert_model_with_tokenizer/saved_tokenizer")

# Define the model
class BertRegressor(nn.Module):
    def __init__(self):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained("bert_model_with_tokenizer/base_bert")
        self.dropout = nn.Dropout(0.3)
        self.total_mark_embed = nn.Linear(1, 32)
        self.regressor = nn.Linear(self.bert.config.hidden_size + 32, 1)

    def forward(self, input_ids, attention_mask, total_marks, labels=None):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        total_marks = total_marks.to(device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        total_embed = self.total_mark_embed(total_marks)
        combined = torch.cat((pooled_output, total_embed), dim=1)
        score = self.regressor(combined).squeeze(1)

        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss_fn = nn.MSELoss()
            loss = loss_fn(score, labels)
        return {"loss": loss, "logits": score}

# Load the trained model
try:
    model = BertRegressor().to(device)
    model.load_state_dict(torch.load("bert_model_with_tokenizer/bert_regressor.pth", map_location=device))
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# OCR from image
def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        result = ocr.ocr(img_array, cls=True)
        text = "\n".join([line[1][0] for line in result[0]]) if result else ""
        return text
    except Exception as e:
        print(f"❌ OCR Error: {e}")
        return ""

# Parse QnA
def clean_ocr_text(text):
    # Fix misrecognized 'g' instead of 'q' at question start
    text = re.sub(r'\bg(\d+)\.', r'q\1.', text, flags=re.IGNORECASE)

    # Fix 'Qs.' -> 'q5.'
    text = re.sub(r'\bQs\.', 'q5.', text, flags=re.IGNORECASE)

    # Fix '{s}' or '{S}' -> '{5}'
    text = re.sub(r'\{[sS]\}', '{5}', text)

    # Fix things like "5}" → "{5}"
    text = re.sub(r'\b(\d)\}', r'{\1}', text)

    # Fix things like "E5}" or "e5}" → "{5}"
    text = re.sub(r'[Ee](\d)\}', r'{\1}', text)

    # Fix double braces: "{{2}" → "{2}", "}}}" → "}"
    text = re.sub(r'\{\{+', '{', text)
    text = re.sub(r'\}\}+', '}', text)
    text = re.sub(r'\bb(\d+)\.', r'6\1.', text, flags=re.IGNORECASE)

    return text


def parse_qa_pairs(text):
    questions = re.split(r'(?i)q\d+\.', text)
    questions = [q.strip() for q in questions if q.strip()]
    
    qa_pairs = []
    for q in questions:
        mark_match = re.search(r'\{+(\d+)[^\}]*\}+', q)
        max_score = int(mark_match.group(1)) if mark_match else 1
        answer = re.sub(r'\{\d+\}$', '', q).strip()
        qa_pairs.append({'answer': answer, 'max_score': max_score})
    
    return qa_pairs

# ✅ Updated feedback function using new Gemini API

# Predict marks
def predict_mark(answer_key, student_answer, max_score):
    input_text = answer_key + " [SEP] " + student_answer
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    max_score_tensor = torch.tensor([[max_score]], dtype=torch.float).to(device)

    with torch.no_grad():
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            total_marks=max_score_tensor
        )
    score = output["logits"].item()
    return min(max(0, score), max_score)
    
@app.route('/upload', methods=['POST'])
def upload_files():
    print("📥 Received POST request to /upload")

    if 'answerKey' not in request.files or 'studentAnswers' not in request.files:
        return jsonify({'error': 'Missing files'}), 400

    answer_key_files = request.files.getlist('answerKey')
    student_answer_files = request.files.getlist('studentAnswers')

    if not answer_key_files or not student_answer_files:
        return jsonify({'error': 'No files selected'}), 400

    try:
        key_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'key')
        student_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'student')
        os.makedirs(key_folder, exist_ok=True)
        os.makedirs(student_folder, exist_ok=True)

        # Save and process answer key files
        answer_key_text = ""
        for file in answer_key_files:
            filename = secure_filename(file.filename)
            path = os.path.join(key_folder, filename)
            file.save(path)
            raw_text = extract_text_from_image(path)
            cleaned_text = clean_ocr_text(raw_text)
            answer_key_text += cleaned_text + "\n"

        rect =rectify_answer_key_ocr(answer_key_text)
        answer_key_data = parse_qa_pairs(rect)
        print(f"📊 Parsed {len(answer_key_data)} questions from answer key")

        # Save and process student answer files
        student_answer_text = ""
        for file in student_answer_files:
            filename = secure_filename(file.filename)
            path = os.path.join(student_folder, filename)
            file.save(path)
            raw_text = extract_text_from_image(path)
            cleaned_text = clean_ocr_text(raw_text)
            student_answer_text += cleaned_text + "\n"
        stud = rectify_student_ocr(student_answer_text)
        student_answer_data = parse_qa_pairs(stud)
        print(f"📊 Parsed {len(student_answer_data)} questions from student answers")

        if len(answer_key_data) != len(student_answer_data):
            return jsonify({
                'error': f'Question count mismatch. Key: {len(answer_key_data)}, Student: {len(student_answer_data)}'
            }), 400

        detailed_scores = {}
        total_score = 0
        total_marks = 0
        feedbacks = []
        all_answers = ""

        for i in range(len(answer_key_data)):
            q_num = i + 1
            max_score = answer_key_data[i]['max_score']
            key_answer = answer_key_data[i]['answer']
            student_answer = student_answer_data[i]['answer']

            print(f"🔍 Scoring Q{q_num} - Max Score: {max_score}")
            print(f"🔑 Key Answer: {key_answer}")
            print(f"📝 Student Answer: {student_answer}")

            predicted_score = predict_mark(key_answer, student_answer, max_score)
            print(f"✅ Predicted score: {predicted_score}")

            # feedback = get_feedback_from_gemini(key_answer, student_answer)

            # feedbacks.append(f"Q{q_num}: {feedback}")
            all_answers += f"Q{q_num} - Key Answer: {key_answer}\nStudent Answer: {student_answer}\n\n"

            detailed_scores[f"Q{q_num}"] = {
                'score': round(predicted_score, 2),
                'max_score': max_score,
                'key_answer': key_answer,
                'student_answer': student_answer,

            }

            total_score += predicted_score
            total_marks += max_score

        overall_feedback = get_feedback_from_gemini(all_answers)

        response = {
            'total_score': round(total_score, 2),
            'total_marks': total_marks,
            'percentage': round((total_score / total_marks) * 100, 2) if total_marks > 0 else 0,
            'overall_feedback':overall_feedback,
            'detailed_scores': detailed_scores
        }

        return jsonify(response)

    except Exception as e:
        print(f"❌ Error processing files: {e}")
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
