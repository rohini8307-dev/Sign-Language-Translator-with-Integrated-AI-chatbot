# Sign-Language-Translator-with-Integrated-AI-chatbot

### *Real-Time Sign Gesture → Text → Sentence → AI Response System*

**Custom Pipeline • End-to-End Engineered • No Paid APIs**

## ABSTRACT

This project implements a real-time **Sign Gesture Translation System** that converts continuous sign gestures into **letters**, forms **meaningful sentences**, and forwards them to an integrated **AI chatbot** which replies in both **text and voice**.

Everything is built from scratch — dataset collection, landmark extraction, MLP model, inference engine, sentence builder, and a complete FastAPI-based web interface.
No pre-trained gesture models or commercial APIs are used.

##PIPELINE:##

1. **Sign Gesture Input** (webcam)
2. **21-Point Landmark Extraction**
3. **Custom MLP Classification** (letter-level)
4. **Sentence Builder** (buffer logic)
5. **AI Chatbot Response** (LLM)
6. **Text + Voice Output**

 <img width="2195" height="691" alt="Untitled design (3)" src="https://github.com/user-attachments/assets/8864b032-9d0f-4581-a19a-d42d8282f18c" />

## TECHNICAL ARCHITECTURE 

### **1. Dataset Creation**

* Custom dataset captured using webcam
* Hundreds of sign gesture frames
* Pre-filtering and clean labelling
  
![ASL_Alphabet](https://github.com/user-attachments/assets/5b64dcbf-2071-4959-b483-22e6afb8ee53)


### **2. Landmark Extraction**

* MediaPipe Hands used to extract **21 landmark coordinates**
* Normalization applied to scale across different hand sizes/distances
* Converts images into structured 42-dimension vectors

### **3. Sign Classification Model (MLP)**

* Input: 42 normalized landmark points
* Optimized MLP architecture for low-latency inference
* Softmax output for alphabet classes (A-Z)

### **4. Sentence Builder**

Transforms continuous single-letter predictions into readable text:

* Character buffer
* Smart spacing
* Delete logic
* Word & sentence formation rules

### **5. AI Chatbot Module**

* Final sentence fed to an LLM
* Generates natural-language reply
* Reply converted into **speech** using TTS

### **6. Web Interface**

* Camera stream
* Real-time prediction overlay
* Chatbot reply panel
* Simple, clean UI

<img width="1691" height="826" alt="Screenshot 2025-11-19 151728" src="https://github.com/user-attachments/assets/e01718ae-ce7e-4cae-96a9-5aebfd0d63dc" />


## CORE FEATURES

*  **Real-time sign gesture recognition**
*  **Accurate MLP classifier trained on custom data**
*  **Pure landmark-based machine learning pipeline**
*  **Integrated AI chatbot with text + voice output**
*  **FastAPI backend with a lightweight frontend**
* **Modular, industry-standard code structure**

## DEMO

<img width="1860" height="702" alt="Screenshot 2025-11-19 225713" src="https://github.com/user-attachments/assets/7450a950-d8fc-4bc7-831d-fd6bf0521f58" />

<img width="1864" height="699" alt="Screenshot 2025-11-20 000150" src="https://github.com/user-attachments/assets/98cdeaae-6de4-4cea-90a2-f6ec24ed8399" />
