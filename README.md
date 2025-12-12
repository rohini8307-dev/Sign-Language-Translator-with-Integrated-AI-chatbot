# Sign Language Translator with Integrated AI chatbot

### *Real-Time Sign Gesture → Text → Sentence → AI Response System*

## ABSTRACT

This project implements a real-time **Sign Gesture Translation System** that converts continuous sign gestures into **letters** and then forms them into a **sentence**, and forwards them to an integrated **AI chatbot** which replies in both **text and voice**.
The project is completely built from scratch including the  dataset collection, landmark extraction, MLP model, inference engine, sentence builder, and a complete FastAPI-based web interface.

## PIPELINE:

1. **Sign Gesture Input** 
2. **21-Point Landmark Extraction**
3. **Custom MLP Classification** 
4. **Sentence Builder** 
5. **AI Chatbot Response** 
6. **Text + Voice Output**
   

 <img width="2195" height="691" alt="Untitled design (3)" src="https://github.com/user-attachments/assets/8864b032-9d0f-4581-a19a-d42d8282f18c" />

## TECHNICAL ARCHITECTURE 

### **1. Dataset Creation**

* Custom dataset captured using webcam
* Each label has 200 images 
* Every image is preprocessed to a resolution of 300 × 300 pixels.
  
  
![ASL_Alphabet](https://github.com/user-attachments/assets/5b64dcbf-2071-4959-b483-22e6afb8ee53)


### **2. Landmark Extraction**

* MediaPipe Hands used to extract **21 landmark coordinates**
* Normalization applied to scale across different hand sizes
* Converts images each into structured 42-dimension vectors

### **3. Sign Classification Model (MLP)**

* Input: 42 normalized landmark points
* Optimized MLP architecture for low-latency inference
* Softmax output for alphabet classes (A-Z)

### **4. Sentence Builder**

Transforms continuous single-letter predictions into readable text with character buffer, Smart spacing, and delete logic.

### **5. AI Chatbot Module**

* Final sentence fed to the LLM (Mistral powered by Ollama)
* Generates natural-language reply
* Reply converted into **speech** using TTS

### **6. Web Interface**

* Camera stream
* Real-time prediction overlay
* Chatbot reply panel
* Simple and clean UI
  

<img width="1691" height="826" alt="Screenshot 2025-11-19 151728" src="https://github.com/user-attachments/assets/e01718ae-ce7e-4cae-96a9-5aebfd0d63dc" />

## TECH STACK

* Tensorflow / Keras
* Mediapipe
* OpenCV
* Ollama : Mistral
* pyttsx3
* Flask
* HTML
* CSS
* Javascript

## CORE FEATURES

*  **Real-time sign gesture recognition**
*  **Accurate MLP classifier trained on custom data**
*  **Pure landmark-based deep learning pipeline**
*  **Integrated AI chatbot with text + voice output**
*  **FastAPI backend with a lightweight frontend**

## DEMO

<img width="1860" height="702" alt="Screenshot 2025-11-19 225713" src="https://github.com/user-attachments/assets/7450a950-d8fc-4bc7-831d-fd6bf0521f58" />


<img width="1864" height="699" alt="Screenshot 2025-11-20 000150" src="https://github.com/user-attachments/assets/98cdeaae-6de4-4cea-90a2-f6ec24ed8399" />
