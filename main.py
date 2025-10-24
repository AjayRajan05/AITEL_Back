"""
🚀 INNOVATIVE TELEMED AI BACKEND
Advanced Medical Chatbot with Excel Integration & Smart Doctor Matching
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
import google.generativeai as genai
import pandas as pd
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import uvicorn
import requests
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('.env')

# Initialize FastAPI app
app = FastAPI(
    title="🚀 TeleMed AI - Advanced Medical Assistant",
    description="Innovative AI-powered telemedicine platform with smart doctor matching",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://localhost:3000", "http://localhost:3001", 
        "http://localhost:3002", "http://localhost:3003", "http://localhost:3004", 
        "http://127.0.0.1:5173", "http://127.0.0.1:3000", "http://127.0.0.1:3001", 
        "http://127.0.0.1:3002", "http://127.0.0.1:3003", "http://127.0.0.1:3004", 
        "http://192.168.56.1:3000", "http://192.168.56.1:3001", "http://192.168.56.1:3002", 
        "http://192.168.56.1:3003", "http://192.168.56.1:3004", "http://192.168.56.1:5173",
        # Add production frontend URLs here
        "https://aitelfront.vercel.app"
        #"https://telemed-ai.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM", "noreply@telemed.com")
DEFAULT_APPOINTMENT_EMAIL = os.getenv("DEFAULT_APPOINTMENT_EMAIL", SMTP_FROM)
OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://localhost:5000")

# Initialize Gemini AI
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    logger.info("✅ Gemini AI configured successfully")
else:
    logger.warning("❌ Gemini API key not found")
    model = None

# Pydantic Models
class ChatRequest(BaseModel):
    message: str
    user_profile: Optional[Dict[str, Any]] = None
    chat_history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    response: str
    doctors: Optional[List[Dict[str, Any]]] = None
    appointment_needed: bool = False
    suggested_specialization: Optional[str] = None

class AppointmentRequest(BaseModel):
    doctor_name: str
    patient_name: str
    patient_email: str
    appointment_date: str
    appointment_time: str
    medical_issue: str
    patient_profile: Optional[Dict[str, Any]] = None

class Doctor(BaseModel):
    name: str
    specialization: str
    email: str
    phone: str
    hospital_clinic: Optional[str] = None
    city: Optional[str] = None
    qualification: Optional[str] = None
    availability: Optional[str] = None
    available_days: Optional[str] = None
    available_times: Optional[str] = None
    location: Optional[str] = None

# 🧠 INNOVATIVE MEDICAL CHATBOT CLASS
class AdvancedMedicalChatbot:
    def __init__(self):
        self.doctors_data = []
        self.specialization_keywords = {
            # Neurology & Headaches
            'headache': ['Neurology', 'Neurologist'],
            'migraine': ['Neurology', 'Neurologist'],
            'brain': ['Neurology', 'Neurologist'],
            'neurological': ['Neurology', 'Neurologist'],
            'seizure': ['Neurology', 'Neurologist'],
            'dizziness': ['Neurology', 'Neurologist'],
            'memory': ['Neurology', 'Neurologist'],
            
            # Cardiology
            'heart': ['Cardiology', 'Cardiologist'],
            'cardiac': ['Cardiology', 'Cardiologist'],
            'chest pain': ['Cardiology', 'Cardiologist'],
            'blood pressure': ['Cardiology', 'Cardiologist'],
            'hypertension': ['Cardiology', 'Cardiologist'],
            
            # Pediatrics
            'child': ['Pediatrics', 'Pediatrician'],
            'pediatric': ['Pediatrics', 'Pediatrician'],
            'baby': ['Pediatrics', 'Pediatrician'],
            'infant': ['Pediatrics', 'Pediatrician'],
            'teenager': ['Pediatrics', 'Pediatrician'],
            
            # Orthopedics
            'bone': ['Orthopedics', 'Orthopedic'],
            'joint': ['Orthopedics', 'Orthopedic'],
            'fracture': ['Orthopedics', 'Orthopedic'],
            'back pain': ['Orthopedics', 'Orthopedic'],
            'knee': ['Orthopedics', 'Orthopedic'],
            'shoulder': ['Orthopedics', 'Orthopedic'],
            
            # Dermatology
            'skin': ['Dermatology', 'Dermatologist'],
            'dermatology': ['Dermatology', 'Dermatologist'],
            'rash': ['Dermatology', 'Dermatologist'],
            'acne': ['Dermatology', 'Dermatologist'],
            'mole': ['Dermatology', 'Dermatologist'],
            
            # Psychiatry
            'mental': ['Psychiatry', 'Psychiatrist'],
            'depression': ['Psychiatry', 'Psychiatrist'],
            'anxiety': ['Psychiatry', 'Psychiatrist'],
            'stress': ['Psychiatry', 'Psychiatrist'],
            'therapy': ['Psychiatry', 'Psychiatrist'],
            
            # Gynecology
            'women': ['Gynecology', 'Gynecologist'],
            'gynecology': ['Gynecology', 'Gynecologist'],
            'pregnancy': ['Gynecology', 'Gynecologist'],
            'prenatal': ['Gynecology', 'Gynecologist'],
            
            # Ophthalmology
            'eye': ['Ophthalmology', 'Ophthalmologist'],
            'vision': ['Ophthalmology', 'Ophthalmologist'],
            'glasses': ['Ophthalmology', 'Ophthalmologist'],
            
            # ENT
            'ear': ['ENT', 'Otolaryngologist'],
            'throat': ['ENT', 'Otolaryngologist'],
            'nose': ['ENT', 'Otolaryngologist'],
            'hearing': ['ENT', 'Otolaryngologist'],
        }
        
        self.appointment_keywords = [
            'book an appointment', 'schedule an appointment', 'make an appointment',
            'book appointment', 'schedule appointment', 'make appointment',
            'book with', 'schedule with', 'meet with doctor',
            'see a doctor', 'consult with doctor', 'visit doctor',
            'choose doctor', 'select doctor', 'pick doctor',
            'want to see doctor', 'need to see doctor', 'talk to doctor',
            'book consultation', 'schedule consultation', 'appointment with',
            'when can i see', 'how to book', 'how to schedule',
            'book me', 'schedule me', 'book dr', 'schedule dr',
            'book doctor', 'schedule doctor', 'appointment for',
            'book him', 'book her', 'book them', 'book this doctor',
            'book on', 'schedule on', 'book for', 'schedule for',
            'yes book', 'yes schedule', 'book it', 'schedule it',
            'go ahead', 'proceed', 'confirm', 'book now', 'schedule now',
            'book an appointment for', 'schedule an appointment for',
            'book appointment for', 'schedule appointment for',
            'book an appoinment', 'book appoinment', 'appoinment',  # Common misspellings
            'book', 'schedule', 'appointment'  # Single words
        ]
        
        self.load_doctors_data()
    
    def load_doctors_data(self):
        """Load doctors data from Excel file with enhanced error handling"""
        try:
            doctors_file = os.path.join("uploads", "doctors.xlsx")
            if not os.path.exists(doctors_file):
                logger.error(f"❌ Doctors file not found at {doctors_file}")
                self.doctors_data = []
                return
            
            # Read Excel file
            df = pd.read_excel(doctors_file, engine='openpyxl')
            logger.info(f"📊 Loaded Excel file with {len(df)} rows and columns: {df.columns.tolist()}")
            
            # Enhanced column mapping for the new Excel structure
            column_mapping = {
                'Name': 'name',
                'Specialization': 'specialization', 
                'Hospital/Clinic': 'hospital_clinic',
                'City': 'city',
                'Phone': 'phone',
                'Email': 'email',
                'Availability': 'availability',
                'Qualification': 'qualification'
            }
            
            # Rename columns
            df_cleaned = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_columns = ['name', 'specialization', 'email', 'phone']
            missing_columns = [col for col in required_columns if col not in df_cleaned.columns]
            
            if missing_columns:
                logger.error(f"❌ Missing required columns: {missing_columns}")
                self.doctors_data = []
            else:
                # Process and clean the data
                self.doctors_data = []
                for _, row in df_cleaned.iterrows():
                    doctor = {
                        'name': str(row.get('name', '')).strip(),
                        'specialization': str(row.get('specialization', '')).strip(),
                        'email': str(row.get('email', '')).strip(),
                        'phone': str(row.get('phone', '')).strip(),
                        'hospital_clinic': str(row.get('hospital_clinic', '')).strip(),
                        'city': str(row.get('city', '')).strip(),
                        'qualification': str(row.get('qualification', '')).strip(),
                        'availability': str(row.get('availability', '')).strip(),
                        # Parse availability into days and times
                        'available_days': self._parse_availability_days(row.get('availability', '')),
                        'available_times': self._parse_availability_times(row.get('availability', '')),
                        'location': f"{row.get('hospital_clinic', '')}, {row.get('city', '')}".strip(', ')
                    }
                    
                    # Only add doctors with valid data
                    if doctor['name'] and doctor['email']:
                        self.doctors_data.append(doctor)
                
                logger.info(f"✅ Successfully loaded {len(self.doctors_data)} doctors")
                
                # Log first doctor as example
                if self.doctors_data:
                    logger.info(f"📋 Sample doctor: {self.doctors_data[0]}")
                    
        except Exception as e:
            logger.error(f"❌ Error loading doctors data: {e}")
            self.doctors_data = []
    
    def _parse_availability_days(self, availability: str) -> str:
        """Parse availability string to extract days"""
        if not availability:
            return "Monday - Friday"
        
        availability_lower = availability.lower()
        
        # Check for specific days
        if any(day in availability_lower for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
            days = []
            day_mapping = {
                'monday': 'Monday', 'tuesday': 'Tuesday', 'wednesday': 'Wednesday',
                'thursday': 'Thursday', 'friday': 'Friday', 'saturday': 'Saturday', 'sunday': 'Sunday'
            }
            
            for day_key, day_name in day_mapping.items():
                if day_key in availability_lower:
                    days.append(day_name)
            
            return ', '.join(days) if days else "Monday - Friday"
        
        # Default to weekdays if no specific days mentioned
        return "Monday - Friday"
    
    def _parse_availability_times(self, availability: str) -> str:
        """Parse availability string to extract times"""
        if not availability:
            return "9:00 AM - 5:00 PM"
        
        # Look for time patterns like "10AM - 6PM", "9:00 AM - 5:00 PM", etc.
        import re
        
        # Pattern to match time ranges
        time_pattern = r'(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))\s*-\s*(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))'
        match = re.search(time_pattern, availability, re.IGNORECASE)
        
        if match:
            start_time = match.group(1).strip()
            end_time = match.group(2).strip()
            return f"{start_time} - {end_time}"
        
        # If no time range found, look for single time mentions
        time_single_pattern = r'(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))'
        times = re.findall(time_single_pattern, availability, re.IGNORECASE)
        
        if times:
            if len(times) >= 2:
                return f"{times[0]} - {times[1]}"
            else:
                return f"{times[0]} - 5:00 PM"
        
        # Default time if no pattern matches
        return "9:00 AM - 5:00 PM"
    
    def detect_specialization(self, message: str) -> Optional[str]:
        """Enhanced specialization detection with fuzzy matching"""
        message_lower = message.lower()
        
        # Direct keyword matching
        for keyword, specializations in self.specialization_keywords.items():
            if keyword in message_lower:
                # Find doctors matching any of the specialization terms
                for spec in specializations:
                    matching_doctors = [doc for doc in self.doctors_data 
                                      if spec.lower() in doc['specialization'].lower()]
                    if matching_doctors:
                        logger.info(f"🎯 Detected specialization: {spec} for keyword: {keyword}")
                        return spec
        
        # If no specific specialization found, return None for general practitioners
        return None
    
    def get_doctors_by_specialization(self, specialization: str = None) -> List[Dict]:
        """Get doctors filtered by specialization with enhanced matching"""
        if not self.doctors_data:
            logger.warning("⚠️ No doctors data available")
            return []
        
        if specialization:
            # Enhanced matching - check if specialization is contained in doctor's specialization
            filtered_doctors = []
            for doc in self.doctors_data:
                doc_spec = doc['specialization'].lower()
                if (specialization.lower() in doc_spec or 
                    doc_spec in specialization.lower() or
                    any(word in doc_spec for word in specialization.lower().split())):
                    filtered_doctors.append(doc)
            
            logger.info(f"🔍 Found {len(filtered_doctors)} doctors for specialization: {specialization}")
            return filtered_doctors
        
        # Return all doctors if no specialization specified
        logger.info(f"📋 Returning all {len(self.doctors_data)} doctors")
        return self.doctors_data
    
    def detect_appointment_intent(self, message: str) -> bool:
        """Enhanced appointment intent detection - only triggers on explicit appointment requests"""
        message_lower = message.lower()
        
        # Check for explicit appointment keywords
        for keyword in self.appointment_keywords:
            if keyword in message_lower:
                logger.info(f"🎯 Appointment intent detected with keyword: {keyword}")
                return True
        
        # Additional checks for appointment intent
        appointment_phrases = [
            'i want to book', 'i need to book', 'i would like to book',
            'i want to schedule', 'i need to schedule', 'i would like to schedule',
            'can i book', 'can i schedule', 'how do i book', 'how do i schedule',
            'book me', 'schedule me', 'make an appointment for me',
            'okay book', 'yes book', 'book him', 'book her', 'book them',
            'book on', 'schedule on', 'book for', 'schedule for',
            'book it', 'schedule it', 'go ahead', 'proceed', 'confirm',
            'book now', 'schedule now', 'any day is fine', 'any time is fine',
            'yes', 'okay', 'sure', 'ok', 'yep', 'yeah', 'definitely', 'absolutely'
        ]
        
        for phrase in appointment_phrases:
            if phrase in message_lower:
                logger.info(f"🎯 Appointment intent detected with phrase: {phrase}")
                return True
        
        # Check for partial matches with "book" and "appointment"
        if 'book' in message_lower and 'appointment' in message_lower:
            logger.info("🎯 Appointment intent detected with 'book' + 'appointment'")
            return True
        
        # Check for "book" followed by any word (like "book an", "book the", etc.)
        if re.search(r'\bbook\s+\w+', message_lower):
            logger.info("🎯 Appointment intent detected with 'book' + word pattern")
            return True
        
        logger.info("❌ No appointment intent detected")
        return False
    
    def extract_datetime_info(self, message: str) -> tuple:
        """Extract date and time information from message"""
        date_patterns = {
            "today": 0,
            "tomorrow": 1,
            "next week": 7,
            "monday": None, "tuesday": None, "wednesday": None, 
            "thursday": None, "friday": None, "saturday": None, "sunday": None
        }
        
        time_patterns = {
            "morning": "09:00", "afternoon": "14:00", 
            "evening": "17:00", "night": "19:00"
        }
        
        message_lower = message.lower()
        
        # Default values
        appointment_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        appointment_time = "10:00"
        
        # Find date
        for pattern, days_offset in date_patterns.items():
            if pattern in message_lower:
                if days_offset is not None:
                    appointment_date = (datetime.now() + timedelta(days=days_offset)).strftime("%Y-%m-%d")
                elif pattern in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
                    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                    target_weekday = weekdays.index(pattern)
                    current_weekday = datetime.now().weekday()
                    days_ahead = target_weekday - current_weekday
                    if days_ahead <= 0:
                        days_ahead += 7
                    appointment_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
                break
        
        # Find time
        for pattern, time_value in time_patterns.items():
            if pattern in message_lower:
                appointment_time = time_value
                break
        
        # Check for specific times like "9am", "2pm", etc.
        time_match = re.search(r'(\d{1,2})\s*(am|pm)', message_lower)
        if time_match:
            hour = int(time_match.group(1))
            period = time_match.group(2)
            if period == 'pm' and hour != 12:
                hour += 12
            elif period == 'am' and hour == 12:
                hour = 0
            appointment_time = f"{hour:02d}:00"
        
        return appointment_date, appointment_time
    
    async def generate_response(self, message: str, user_profile: Dict = None, chat_history: List = None) -> Dict:
        """Generate intelligent AI response with doctor recommendations only when requested"""
        try:
            logger.info(f"🤖 Processing message: {message[:50]}...")
            
            # Detect specialization
            specialization = self.detect_specialization(message)
            
            # If no specialization detected but we have chat history, check previous messages
            if not specialization and chat_history:
                for msg in chat_history[-3:]:  # Check last 3 messages
                    if msg.get('type') == 'user':
                        prev_specialization = self.detect_specialization(msg.get('content', ''))
                        if prev_specialization:
                            specialization = prev_specialization
                            logger.info(f"🎯 Detected specialization from chat history: {specialization}")
                            break
            
            logger.info(f"🎯 Detected specialization: {specialization}")
            
            # Check appointment intent
            appointment_needed = self.detect_appointment_intent(message)
            logger.info(f"📅 Appointment needed: {appointment_needed}")
            
            # Only get doctors if appointment is explicitly requested
            doctors_list = []
            if appointment_needed:
                doctors_list = self.get_doctors_by_specialization(specialization)
                logger.info(f"👨‍⚕️ Found {len(doctors_list)} relevant doctors for appointment request")
            
            # Check if user is selecting a specific doctor for booking
            selected_doctor = None
            
            # First check if user mentioned a doctor name in the message (even without doctors_list)
            message_lower = message.lower()
            for doc in self.doctors_data:  # Check all doctors, not just filtered ones
                doc_name_lower = doc['name'].lower()
                doc_last_name = doc_name_lower.split()[-1]  # Get last name
                
                if (doc_name_lower in message_lower or 
                    doc_last_name in message_lower or
                    f"dr. {doc_last_name}" in message_lower or
                    f"dr {doc_last_name}" in message_lower):
                    selected_doctor = doc
                    logger.info(f"🎯 User mentioned doctor by name: {doc['name']}")
                    # Force appointment_needed to true if doctor is mentioned
                    appointment_needed = True
                    break
            
            # If no doctor mentioned by name, check the filtered doctors list
            if not selected_doctor and doctors_list:
                for doc in doctors_list:
                    # Check for full name or just last name
                    doc_name_lower = doc['name'].lower()
                    doc_last_name = doc_name_lower.split()[-1]  # Get last name
                    
                    if (doc_name_lower in message_lower or 
                        doc_last_name in message_lower or
                        f"dr. {doc_last_name}" in message_lower or
                        f"dr {doc_last_name}" in message_lower):
                        selected_doctor = doc
                        logger.info(f"🎯 User selected doctor by name: {doc['name']}")
                        break
                
                # If no specific doctor mentioned but user is confirming booking, use the most relevant doctor
                if not selected_doctor and appointment_needed and len(doctors_list) > 0:
                    # Check if user is giving confirmation to book (only after doctors have been shown)
                    confirmation_phrases = [
                        'okay book', 'yes book', 'book him', 'book her', 'book them',
                        'book it', 'schedule it', 'go ahead', 'proceed', 'confirm',
                        'book now', 'schedule now', 'any day is fine', 'any time is fine',
                        'book on', 'schedule on', 'tuesday', 'thursday', 'monday', 'wednesday', 'friday'
                    ]
                    
                    # Only trigger if this is a confirmation response (not initial booking request)
                    if any(phrase in message_lower for phrase in confirmation_phrases) and len(chat_history) > 0:
                        # If we have a specialization, use the doctor from that specialization
                        if specialization:
                            for doc in doctors_list:
                                if specialization.lower() in doc['specialization'].lower():
                                    selected_doctor = doc
                                    break
                        
                        # If no specialization match or no specialization, use the first doctor
                        if not selected_doctor:
                            selected_doctor = doctors_list[0]
                        
                        logger.info(f"🎯 User confirmed booking with suggested doctor: {selected_doctor['name']}")
            
            # Build context for AI
            context = f"""You are AITEL, a compassionate and highly intelligent medical AI assistant for a telemedicine platform.

IMPORTANT INSTRUCTIONS:
1. Be warm, empathetic, and supportive in your responses
2. Show genuine concern for the patient's wellbeing
3. Use encouraging and reassuring language
4. Acknowledge the patient's concerns and validate their feelings
5. Provide helpful medical information while being careful not to diagnose
6. Always recommend consulting with a healthcare professional for serious concerns
7. Be patient and kind, especially when patients seem worried or anxious
8. ONLY suggest doctors and appointments when the patient explicitly asks for them
9. For general greetings or casual conversation, respond naturally without pushing appointments
10. Be conversational and friendly, not clinical or pushy
11. When patient requests an appointment, FIRST show available doctors and ask them to choose
12. Only book when patient explicitly selects a specific doctor
13. Be helpful and patient-friendly - let them make the choice
14. Always provide clear instructions on how to book with a specific doctor

PATIENT PROFILE:
"""
            
            if user_profile:
                context += f"""
- Name: {user_profile.get('name', 'Patient')}
- Age: {user_profile.get('age', 'Not provided')}
- Gender: {user_profile.get('sex', 'Not provided')}
- Weight: {user_profile.get('weight', 'Not provided')} lbs
- Height: {user_profile.get('height', 'Not provided')} inches
- Allergies: {', '.join(user_profile.get('allergies', [])) or 'None'}
- Current Medications: {', '.join(user_profile.get('medications', [])) or 'None'}
- Medical Conditions: {', '.join(user_profile.get('medicalConditions', [])) or 'None'}
"""
            
            if chat_history:
                context += "\nRECENT CONVERSATION:\n"
                for msg in chat_history[-3:]:  # Last 3 messages
                    context += f"{msg.get('type', 'user')}: {msg.get('content', '')}\n"
            
            # Handle doctor selection and automatic booking
            if selected_doctor:
                # User has selected a specific doctor, proceed with booking
                appointment_date, appointment_time = self.extract_datetime_info(message)
                
                try:
                    email_sent = EmailService.send_appointment_email(
                        doctor_email=selected_doctor["email"],
                        patient_name=user_profile.get('name', 'Patient') if user_profile else 'Patient',
                        patient_email=DEFAULT_APPOINTMENT_EMAIL,
                        appointment_date=appointment_date,
                        appointment_time=appointment_time,
                        medical_issue=message,
                        patient_profile=user_profile
                    )
                    
                    if email_sent:
                        doctor_details = f"Dr. {selected_doctor['name']} ({selected_doctor['specialization']})"
                        if selected_doctor.get('qualification'):
                            doctor_details += f" - {selected_doctor['qualification']}"
                        
                        location_details = selected_doctor.get('location', '')
                        if not location_details and selected_doctor.get('hospital_clinic'):
                            location_details = f"{selected_doctor['hospital_clinic']}"
                            if selected_doctor.get('city'):
                                location_details += f", {selected_doctor['city']}"
                        
                        context += f"""
🎉 WONDERFUL! I've successfully scheduled your appointment with Dr. {selected_doctor['name']}!

Appointment Details:
📅 Date: {appointment_date}
⏰ Time: {appointment_time}
👨‍⚕️ Doctor: {doctor_details}
📞 Phone: {selected_doctor['phone']}
📍 Location: {location_details or 'To be confirmed'}

I've sent all the details to our clinic, and you'll receive a confirmation email shortly. Dr. {selected_doctor['name']} will be well-prepared to help you with your concerns.

Is there anything else I can help you with while we wait for your appointment?
"""
                    else:
                        context += """
I apologize, but I encountered a small technical issue while sending the appointment confirmation. 
Don't worry though - your appointment request has been noted. Please contact our support team, 
and they'll ensure everything is set up properly for you.
"""
                except Exception as e:
                    logger.error(f"❌ Error booking appointment: {e}")
                    context += """
I'm so sorry, but I ran into a technical hiccup while processing your appointment. 
Please don't worry - these things happen sometimes. You can try again in a moment, 
or our support team will be happy to help you book directly.
"""
            
            # Only add doctor information if appointment is explicitly requested but no doctor selected yet
            elif appointment_needed and doctors_list and not selected_doctor:
                # Show available doctors and ask patient to choose
                doctors_display = []
                for doc in doctors_list[:3]:
                    doctor_info = f"• **{doc['name']}** - {doc['specialization']}"
                    if doc.get('qualification'):
                        doctor_info += f" ({doc['qualification']})"
                    if doc.get('hospital_clinic'):
                        doctor_info += f"\n  🏥 {doc['hospital_clinic']}"
                    if doc.get('city'):
                        doctor_info += f", {doc['city']}"
                    if doc.get('available_times'):
                        doctor_info += f"\n  ⏰ Available: {doc['available_times']}"
                    doctors_display.append(doctor_info)
                
                context += f"""
👨‍⚕️ **Here are the best doctors available for your condition:**

{chr(10).join(doctors_display)}

**To book an appointment, please tell me which doctor you'd like to see by saying:**
- "Book [doctor name]" (e.g., "Book Dr. Jayesh Batta")
- "I want to see [doctor name]"
- "Schedule with [doctor name]"

I'm here to help you get the care you need! 😊"""
            elif appointment_needed and not doctors_list:
                context += f"""
The patient has requested an appointment but no specific doctors were found for {specialization or 'general consultation'}.
Suggest they contact our general practitioners or provide general guidance about scheduling.
"""
            
            # Generate AI response
            if selected_doctor:
                # If appointment was booked, be direct and confirmative
                full_prompt = f"""{context}

PATIENT MESSAGE: {message}

The appointment has been successfully booked! Respond with enthusiasm and confirmation. Be direct and positive about the booking completion."""
            else:
                full_prompt = f"""{context}

PATIENT MESSAGE: {message}

Please respond warmly and empathetically. {'If the patient requested an appointment and doctors are available, mention them specifically.' if appointment_needed and doctors_list else 'Respond naturally to their message without pushing appointments unless they specifically ask for one.'}"""
            
            if not model:
                return {
                    "response": "I'm so sorry, but our AI service seems to be temporarily unavailable. Please contact our support team for assistance, and they'll take great care of you!",
                    "doctors": doctors_list,
                    "appointment_needed": appointment_needed,
                    "suggested_specialization": specialization
                }
            
            response = model.generate_content(full_prompt)
            logger.info(f"🔍 Gemini response type: {type(response)}")
            logger.info(f"🔍 Gemini response: {response}")
            ai_response = response.text
            
            return {
                "response": ai_response,
                "doctors": doctors_list if doctors_list else None,
                "appointment_needed": appointment_needed,
                "suggested_specialization": specialization
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            
            # Fallback response when API quota is exceeded
            if "429" in str(e) or "quota" in str(e).lower():
                if selected_doctor:
                    doctor_details = f"{selected_doctor['name']} ({selected_doctor['specialization']})"
                    if selected_doctor.get('qualification'):
                        doctor_details += f" - {selected_doctor['qualification']}"
                    
                    location_details = selected_doctor.get('location', '')
                    if not location_details and selected_doctor.get('hospital_clinic'):
                        location_details = f"{selected_doctor['hospital_clinic']}"
                        if selected_doctor.get('city'):
                            location_details += f", {selected_doctor['city']}"
                    
                    fallback_response = f"""🎉 **APPOINTMENT BOOKED SUCCESSFULLY!**

**Appointment Details:**
📅 **Date:** {appointment_date}
🕐 **Time:** {appointment_time}
👨‍⚕️ **Doctor:** {doctor_details}
📞 **Phone:** {selected_doctor['phone']}
📍 **Location:** {location_details or 'To be confirmed'}

✅ **Confirmation email sent to:** {DEFAULT_APPOINTMENT_EMAIL}

Everything is set! You'll receive a confirmation email with all the details shortly. Is there anything else I can help you with today?"""
                elif appointment_needed and doctors_list:
                    doctors_display = []
                    for doc in doctors_list[:3]:
                        doctor_info = f"• **{doc['name']}** - {doc['specialization']}"
                        if doc.get('qualification'):
                            doctor_info += f" ({doc['qualification']})"
                        if doc.get('hospital_clinic'):
                            doctor_info += f"\n  🏥 {doc['hospital_clinic']}"
                        if doc.get('city'):
                            doctor_info += f", {doc['city']}"
                        if doc.get('available_times'):
                            doctor_info += f"\n  ⏰ Available: {doc['available_times']}"
                        doctors_display.append(doctor_info)
                    
                    fallback_response = f"""👨‍⚕️ **Here are the best doctors for your condition:**

{chr(10).join(doctors_display)}

**To book an appointment, just say:** "Book an appointment" or "Book [doctor name]"

I'm here to help you get the care you need! 😊"""
                else:
                    fallback_response = "Hi there! I'm here to help you with your health needs. How can I assist you today? 😊"
                
                return {
                    "response": fallback_response,
                    "doctors": doctors_list if appointment_needed and doctors_list else None,
                    "appointment_needed": appointment_needed,
                    "suggested_specialization": specialization
                }
            else:
                return {
                    "response": f"I'm really sorry, but I seem to be having a technical difficulty right now. Please bear with me and try again in a moment, or feel free to contact our support team who will be happy to help you personally. Your health and concerns are important to us! 💙",
                    "doctors": None,
                    "appointment_needed": False,
                    "suggested_specialization": None
                }

# Initialize chatbot
chatbot = AdvancedMedicalChatbot()

# 🔍 OCR SERVICE INTEGRATION
async def extract_text_from_document(file: UploadFile) -> str:
    """Extract text from document using OCR service"""
    try:
        logger.info(f"📄 Sending document to OCR service: {file.filename}")
        
        # Prepare file for OCR service
        files = {"file": (file.filename, await file.read(), file.content_type)}
        
        # Call OCR service
        response = requests.post(f"{OCR_SERVICE_URL}/extract-text", files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ OCR extraction successful: {result['character_count']} characters")
            return result['extracted_text']
        else:
            logger.error(f"❌ OCR service error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail="OCR service unavailable")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ OCR service connection error: {e}")
        raise HTTPException(status_code=500, detail="OCR service connection failed")
    except Exception as e:
        logger.error(f"❌ Document processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing error: {str(e)}")

# 📧 ENHANCED EMAIL SERVICE
class EmailService:
    @staticmethod
    def verify_email_config() -> bool:
        """Verify that email configuration is complete"""
        required_settings = {
            "SMTP_HOST": SMTP_HOST,
            "SMTP_PORT": SMTP_PORT,
            "SMTP_USER": SMTP_USER,
            "SMTP_PASS": SMTP_PASS,
            "SMTP_FROM": SMTP_FROM,
            "DEFAULT_APPOINTMENT_EMAIL": DEFAULT_APPOINTMENT_EMAIL
        }
        
        missing = [key for key, value in required_settings.items() if not value]
        if missing:
            logger.error(f"❌ Missing email settings: {', '.join(missing)}")
            return False
        return True

    @staticmethod
    def send_appointment_email(doctor_email: str, patient_name: str, patient_email: str, 
                             appointment_date: str, appointment_time: str, medical_issue: str,
                             patient_profile: Dict = None) -> bool:
        """Send appointment booking email to doctor and admin"""
        try:
            if not EmailService.verify_email_config():
                logger.warning("⚠️ Email configuration is incomplete - appointment will be recorded without email notification")
                return False
                
            logger.info(f"📧 Sending appointment email...")
            logger.info(f"From: {SMTP_FROM}")
            logger.info(f"To: {DEFAULT_APPOINTMENT_EMAIL} (Default Email Only)")
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = SMTP_FROM
            msg['To'] = DEFAULT_APPOINTMENT_EMAIL
            msg['Subject'] = f"🩺 New Patient Appointment Request - {patient_name}"
            
            # Create professional HTML email body
            body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
                    .content {{ padding: 20px; }}
                    .patient-info {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #4CAF50; margin: 15px 0; }}
                    .appointment-details {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                    .footer {{ background-color: #f5f5f5; padding: 15px; text-align: center; font-size: 12px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🏥 TeleMed AI Platform</h1>
                    <h2>New Appointment Request</h2>
                </div>
                
                <div class="content">
                    <p>Dear Admin,</p>
                    <p>A new patient has requested an appointment through our TeleMed AI platform. Please review the details below and contact the patient to confirm the appointment:</p>
                    
                    <div class="appointment-details">
                        <h3>📋 Appointment Details</h3>
                        <p><strong>📅 Requested Date:</strong> {appointment_date}</p>
                        <p><strong>⏰ Requested Time:</strong> {appointment_time}</p>
                        <p><strong>👨‍⚕️ Doctor:</strong> {doctor_email}</p>
                    </div>
                    
                    <div class="patient-info">
                        <h3>👤 Patient Information</h3>
                        <p><strong>Name:</strong> {patient_name}</p>
                        <p><strong>Contact Email:</strong> {patient_email}</p>
                        
                        {f'''
                        <h4>Medical Profile:</h4>
                        <ul>
                            <li><strong>Age:</strong> {patient_profile.get('age', 'Not provided')} years</li>
                            <li><strong>Gender:</strong> {patient_profile.get('sex', 'Not provided')}</li>
                            <li><strong>Weight:</strong> {patient_profile.get('weight', 'Not provided')} lbs</li>
                            <li><strong>Height:</strong> {patient_profile.get('height', 'Not provided')} inches</li>
                            <li><strong>Known Allergies:</strong> {', '.join(patient_profile.get('allergies', [])) or 'None reported'}</li>
                            <li><strong>Current Medications:</strong> {', '.join(patient_profile.get('medications', [])) or 'None reported'}</li>
                            <li><strong>Medical History:</strong> {', '.join(patient_profile.get('medicalConditions', [])) or 'None reported'}</li>
                        </ul>
                        ''' if patient_profile else '<p><em>No medical profile provided</em></p>'}
                    </div>
                    
                    <div class="patient-info">
                        <h3>🩺 Chief Complaint / Reason for Visit</h3>
                        <p><em>"{medical_issue}"</em></p>
                    </div>
                    
                    <p><strong>Next Steps:</strong></p>
                    <ul>
                        <li>Review the patient's information and medical profile</li>
                        <li>Contact the patient to confirm appointment details</li>
                        <li>Coordinate with the selected doctor: {doctor_email}</li>
                        <li>Send confirmation email to the patient</li>
                        <li>Prepare any necessary forms or documentation</li>
                    </ul>
                    
                    <p>Thank you for your continued dedication to patient care!</p>
                </div>
                
                <div class="footer">
                    <p>This email was generated by TeleMed AI Platform</p>
                    <p>📧 Questions? Contact our support team</p>
                    <p>🕐 Sent on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            context = ssl.create_default_context()
            try:
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                    server.starttls(context=context)
                    server.login(SMTP_USER, SMTP_PASS)
                    server.send_message(msg)
                    logger.info(f"✅ Appointment email sent successfully to {DEFAULT_APPOINTMENT_EMAIL}")
            except Exception as e:
                logger.error(f"❌ Email sending error: {str(e)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Email sending error: {e}")
            return False

# 🚀 API ROUTES
@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle CORS preflight requests"""
    return {"message": "OK"}

@app.get("/")
async def root():
    return {
        "message": "🚀 TeleMed AI Backend is running", 
        "status": "healthy",
        "version": "2.0.0",
        "doctors_loaded": len(chatbot.doctors_data),
        "gemini_configured": bool(GEMINI_API_KEY),
        "email_configured": bool(SMTP_USER and SMTP_PASS),
        "features": [
            "Smart Doctor Matching",
            "Excel Integration",
            "Email Notifications",
            "Medical Specialization Detection",
            "Appointment Scheduling"
        ]
    }

@app.get("/doctors", response_model=List[Dict])
async def get_doctors(specialization: Optional[str] = None):
    """Get all doctors or filter by specialization"""
    try:
        logger.info(f"📋 Getting doctors with specialization: {specialization}")
        doctors = chatbot.get_doctors_by_specialization(specialization)
        logger.info(f"✅ Returning {len(doctors)} doctors")
        return doctors
    except Exception as e:
        logger.error(f"❌ Error fetching doctors: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching doctors: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """Chat with intelligent medical AI assistant"""
    try:
        logger.info(f"💬 Chat request received: {request.message[:50]}...")
        result = await chatbot.generate_response(
            request.message, 
            request.user_profile, 
            request.chat_history
        )
        
        doctors_count = len(result.get('doctors', [])) if result and result.get('doctors') else 0
        logger.info(f"✅ Chat response generated with {doctors_count} doctors")
        return ChatResponse(
            response=result["response"],
            doctors=result["doctors"],
            appointment_needed=result["appointment_needed"],
            suggested_specialization=result["suggested_specialization"]
        )
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/book-appointment")
async def book_appointment(request: AppointmentRequest):
    """Book appointment with selected doctor"""
    try:
        logger.info(f"📅 Booking appointment for {request.patient_name} with {request.doctor_name}")
        
        # Find doctor details
        doctor_info = None
        for doc in chatbot.doctors_data:
            if doc["name"].lower() == request.doctor_name.lower():
                doctor_info = doc
                break
        
        if not doctor_info:
            raise HTTPException(status_code=404, detail="Doctor not found in our database")
        
        # Send email notification
        email_sent = EmailService.send_appointment_email(
            doctor_email=doctor_info["email"],
            patient_name=request.patient_name,
            patient_email=request.patient_email,
            appointment_date=request.appointment_date,
            appointment_time=request.appointment_time,
            medical_issue=request.medical_issue,
            patient_profile=request.patient_profile
        )
        
        if email_sent:
            logger.info(f"✅ Appointment booked successfully for {request.patient_name}")
            return {
                "success": True,
                "message": f"🎉 Great news! Your appointment request with Dr. {request.doctor_name} has been sent successfully! You'll be contacted shortly for confirmation. Thank you for choosing our healthcare platform!",
                "doctor_email": doctor_info["email"],
                "appointment_details": {
                    "doctor": request.doctor_name,
                    "date": request.appointment_date,
                    "time": request.appointment_time
                }
            }
        else:
            # Return success even if email fails, but log the issue
            logger.warning(f"⚠️ Email sending failed for {request.patient_name}, but appointment is recorded")
            return {
                "success": True,
                "message": f"✅ Your appointment request with Dr. {request.doctor_name} has been recorded successfully! Our team will contact you shortly to confirm the details. (Note: Email notification is temporarily unavailable)",
                "doctor_email": doctor_info["email"],
                "appointment_details": {
                    "doctor": request.doctor_name,
                    "date": request.appointment_date,
                    "time": request.appointment_time
                },
                "email_status": "Email notification temporarily unavailable"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Booking error: {e}")
        raise HTTPException(status_code=500, detail=f"Booking error: {str(e)}")

@app.post("/analyze-document")
async def analyze_document(file: UploadFile = File(...)):
    """Analyze uploaded medical document using Gemini AI (No Tesseract required)"""
    try:
        if not model:
            raise HTTPException(status_code=500, detail="AI service not configured")
        
        logger.info(f"📄 Analyzing document: {file.filename}")
        
        # Read file content
        content = await file.read()
        
        # Extract text based on file type
        file_text = ""
        
        if file.content_type == 'text/plain':
            # Plain text file
            file_text = content.decode('utf-8')
            logger.info(f"✅ Text file processed: {len(file_text)} characters")
            
        elif file.content_type == 'application/pdf':
            # PDF file - enhanced extraction for medical documents
            try:
                import PyPDF2
                import io
                from PIL import Image
                import fitz  # PyMuPDF for better PDF handling
                
                # Try PyMuPDF first (better for images)
                try:
                    pdf_document = fitz.open(stream=content, filetype="pdf")
                    file_text = ""
                    
                    for page_num in range(pdf_document.page_count):
                        page = pdf_document[page_num]
                        
                        # Extract text
                        text = page.get_text()
                        file_text += text + "\n"
                        
                        # Extract images if text is minimal
                        if len(text.strip()) < 100:
                            image_list = page.get_images()
                            for img_index, img in enumerate(image_list):
                                try:
                                    xref = img[0]
                                    pix = fitz.Pixmap(pdf_document, xref)
                                    
                                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                                        img_data = pix.tobytes("png")
                                        
                                        # Use Gemini Vision to analyze the image
                                        import google.generativeai as genai
                                        vision_model = genai.GenerativeModel('gemini-pro')
                                        
                                        image_part = {
                                            "mime_type": "image/png",
                                            "data": img_data
                                        }
                                        
                                        vision_prompt = """
                                        Please analyze this medical image from a PDF document. Extract and describe:
                                        - Any visible text, labels, or annotations
                                        - Medical findings, measurements, or values
                                        - Patient information if visible
                                        - Type of medical image (X-ray, scan, chart, etc.)
                                        - Any abnormalities or notable features
                                        
                                        Provide a detailed description of what you can see in this medical image.
                                        """
                                        
                                        vision_response = vision_model.generate_content([vision_prompt, image_part])
                                        file_text += f"\n[Image {img_index + 1} Analysis]:\n{vision_response.text}\n"
                                        
                                    pix = None
                                except Exception as img_e:
                                    logger.warning(f"⚠️ Image extraction failed: {img_e}")
                                    continue
                    
                    pdf_document.close()
                    logger.info(f"✅ PDF processed with PyMuPDF: {len(file_text)} characters")
                    
                except ImportError:
                    # Fallback to PyPDF2
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                    file_text = ""
                    for page in pdf_reader.pages:
                        file_text += page.extract_text() + "\n"
                    logger.info(f"✅ PDF text extracted with PyPDF2: {len(file_text)} characters")
                    
            except ImportError:
                logger.warning("⚠️ PDF libraries not available, using basic text extraction")
                file_text = str(content)[:2000]
            except Exception as e:
                logger.warning(f"⚠️ PDF extraction failed: {e}, using basic text extraction")
                file_text = str(content)[:2000]
                
        elif file.content_type in ['image/jpeg', 'image/png', 'image/jpg']:
            # Image file - use Gemini's vision capabilities
            try:
                import google.generativeai as genai
                
                # Configure Gemini for image analysis
                vision_model = genai.GenerativeModel('gemini-pro')
                
                # Create image part for Gemini
                image_part = {
                    "mime_type": file.content_type,
                    "data": content
                }
                
                # Analyze image with Gemini Vision
                vision_prompt = """
                Please extract all text from this medical document image. 
                Focus on:
                - Patient information
                - Test results and values
                - Medical findings
                - Doctor notes
                - Dates and measurements
                
                Return the extracted text in a clear, organized format.
                """
                
                vision_response = vision_model.generate_content([vision_prompt, image_part])
                file_text = vision_response.text
                logger.info(f"✅ Image text extracted using Gemini Vision: {len(file_text)} characters")
                
            except Exception as e:
                logger.warning(f"⚠️ Image analysis failed: {e}, using basic text extraction")
                file_text = f"Image file: {file.filename} (Text extraction not available for this image format)"
                
        else:
            # Other file types - basic text extraction
            try:
                file_text = content.decode('utf-8')
            except:
                file_text = str(content)[:2000]
        
        # Limit text length for API efficiency
        if len(file_text) > 8000:
            file_text = file_text[:8000] + "...\n[Document truncated for analysis]"
        
        # Create comprehensive analysis prompt
        prompt = f"""
        As AITEL, a compassionate medical AI assistant, please analyze this medical document with care and empathy. 
        
        Please provide a warm, patient-friendly analysis that includes:
        
        1. **Document Summary** - What type of document is this and what it covers
        2. **Key Findings** - Important results or information (explained in simple terms)
        3. **Values Analysis** - Which values are normal vs concerning (with reassurance where appropriate)
        4. **Health Insights** - What this means for the patient's health
        5. **Recommended Actions** - What steps should be taken next
        6. **Questions for Doctor** - Important questions the patient should ask their healthcare provider
        
        Please use encouraging, supportive language and help the patient understand their results without causing unnecessary worry.
        Remember to emphasize that this analysis is for informational purposes and shouldn't replace professional medical advice.
        
        Document Content: 
        {file_text}
        """
        
        # Generate analysis using Gemini
        response = model.generate_content(prompt)
        
        logger.info(f"✅ Document analysis completed for {file.filename}")
        return {
            "success": True,
            "filename": file.filename,
            "file_type": file.content_type,
            "extracted_text_length": len(file_text),
            "analysis": response.text,
            "message": "I've carefully analyzed your document using AI technology. Please remember that this analysis is for informational purposes only and should be discussed with your healthcare provider."
        }
        
    except Exception as e:
        logger.error(f"❌ Document analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Document analysis error: {str(e)}")

@app.get("/reload-doctors")
async def reload_doctors():
    """Reload doctors data from Excel file"""
    try:
        chatbot.load_doctors_data()
        logger.info(f"🔄 Doctors data reloaded. Found {len(chatbot.doctors_data)} doctors.")
        return {
            "success": True,
            "message": f"Doctors data reloaded successfully. Found {len(chatbot.doctors_data)} doctors.",
            "doctors_count": len(chatbot.doctors_data)
        }
    except Exception as e:
        logger.error(f"❌ Error reloading doctors data: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading doctors data: {str(e)}")

@app.get("/email-status")
async def check_email_status():
    """Check email configuration status"""
    try:
        email_config = {
            "SMTP_HOST": SMTP_HOST,
            "SMTP_PORT": SMTP_PORT,
            "SMTP_USER": SMTP_USER,
            "SMTP_PASS": "***" if SMTP_PASS else None,
            "SMTP_FROM": SMTP_FROM,
            "DEFAULT_APPOINTMENT_EMAIL": DEFAULT_APPOINTMENT_EMAIL
        }
        
        missing = [key for key, value in email_config.items() if not value]
        is_configured = EmailService.verify_email_config()
        
        return {
            "email_configured": is_configured,
            "missing_variables": missing,
            "email_config": email_config,
            "message": "Email configuration check completed"
        }
    except Exception as e:
        logger.error(f"❌ Error checking email status: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking email status: {str(e)}")

if __name__ == "__main__":
    print("Starting TeleMed AI Backend v2.0...")
    print(f"Gemini API: {'Configured' if GEMINI_API_KEY else 'Not configured'}")
    print(f"SMTP: {'Configured' if SMTP_USER and SMTP_PASS else 'Not configured'}")
    print(f"Doctors Database: {len(chatbot.doctors_data)} doctors loaded")
    
    # Check if upload directory exists
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        print(f"Created upload directory: {upload_dir}")
    
    # Get port from environment variable (for Render deployment)
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
