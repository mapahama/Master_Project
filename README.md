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
2. Python version 3.11 or higher is required
   ```bash
   https://www.python.org/downloads/release/python-3110/
3. Python Virtual Environment is required
   ```bash
   # Windows
   python -m venv venv311 
   # Linux/Mac
   python3.11 -m venv venv311
4. Activate Virtual Environment
   ```bash
   # Windows
   venv311\Scripts\activate
   # Linux/Mac
   source venv311/bin/activate
5. Configure VS Code to use venv311
   
    1 ) In VS Code, press Ctrl+Shift+P.
    2 ) Search for “Python: Select Interpreter” and select it.
    3 ) Choose the interpreter from your venv311 folder
       (e.g. .\venv311\Scripts\python.exe on Windows).
6. Install Libraries and Dependencies
   ```bash
   pip install -r requirements.txt
7. TODO
   
