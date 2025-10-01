## Privacy-Preserving Medical Diagnosis with Homomorphic Encryption

This repository contains the implementation of my Master's thesis project on **Classification of homomorphically encrypted data**.  
The project explores how **homomorphic encryption (CKKS scheme)** can be combined with 
**prototype-based machine learning algorithms (GLVQ and GMLVQ)** to build privacy-preserving medical diagnosis systems.

## Overview

The goal of this work is to demonstrate that patient data can remain encrypted at all times during the classification process, 
while still enabling meaningful and explainable medical predictions.  

Three system variants have been implemented:

- **App GLVQ – (Kombination of CKKS & GLVQ with client-side argmin search)**  
  - Encrypted patient vectors are sent to the server.  
  - The server computes encrypted distances to prototypes.  
  - The client decrypts the distances and determines the predicted class.  

- **App GMLVQ - (Kombination of CKKS & GMLVQ with client-side argmin search)**
  - Extends App GLVQ , using relevance matrix.  
  - The server performs encrypted projections and distance calculations.  
  - The client decrypts results and performs class assignment.  

- **App G(M)LVQ sgn Approx - (Kombination of CKKS & G(M)LVQ with fully homomorphic server classification)**  
  - The entire inference, including the **argmin search**, is carried out on the server.  
  - Uses **polynomial approximations (Chebyshev and composite polynomials)** of the Signum Function for encrypted comparisons.  
  - The client only encrypts patient data inputs and decrypts the final class label.

## Technologies

- **Homomorphic encryption**: [TenSEAL](https://github.com/OpenMined/TenSEAL) with CKKS  
- **Machine learning**: GLVQ and GMLVQ (sklearn-lvq)  
- **Backend**: FastAPI (Python)  
- **Frontend**: Streamlit  
- **Dataset**: UCI Heart Disease dataset  
- **Cryptographic foundations**: Ring-LWE, CKKS encoding, polynomial approximations  
 
-----------------------------------------------------------------------------------
## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/mapahama/Master_Project.git
   cd Master_Project
2. Python version 3.11 required (libraries are adjusted to version 3.11)
   ```bash
   https://www.python.org/downloads/release/python-3110/
3. Python Virtual Environment is required
   ```bash
   # Windows
   py -3.11 -m venv venv311
   # Linux/Mac
   python3.11 -m venv venv311
4. Activate Virtual Environment
   ```bash
   # Windows
   venv311\Scripts\activate
   # Linux/Mac
   source venv311/bin/activate
5. Configure VS Code to use venv311
   ```bash
    1) In VS Code, press Ctrl+Shift+P
    2) Search for “Python: Select Interpreter” and select it.
    3) Choose the interpreter from your venv311 folder
       (e.g. .\venv311\Scripts\python.exe on Windows).
6. Install Libraries and Dependencies
   ```bash
   pip install -r requirements.txt
7. Select which App should be used. Navigate in the console to (App GLVQ) or (App GMLVQ) or (App GLVQ sgn Approx)
   ```bash
   #Example: Navigating to App GLVQ
   cd App GLVQ
9. Start the Server 
   ```bash
   uvicorn server:app --reload
   (if You want to run  (App GLVQ sgn Approx), then You have to restart the server after Step 10)
10. Start User Interface (Client)
    ```bash
    1) Open a second terminal
    2) Activate Virtual Environment (Step 4)
    3) Navigate to the App (Step 7)
    4) Run the Streamlit App via the following command in the second terminal:
    streamlit run client.py
    
11. Enter patient data in the form (blood test results, related to Heart Disease risk)
12. Submit the data by clicking the "send" button. The patient data will be encrypted and sent to the ML-Model for classification.
    
13. Wait ~5 seconds for the ML-Model to compute the result
14. The ML model will classify the input as either:
    - healthy (low risk of heart disease), or
    - ill (higher risk of heart disease)
  
15. The result will be visualized in both a table and chart
16. A short explanation will describe why the model chose this classification based on the input features
   
