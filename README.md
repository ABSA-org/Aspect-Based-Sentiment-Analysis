#Aspect-Based Sentiment Analysis (ABSA) Project

#IMPORTANT INSTRUCTIONS:
1.Please use python version 3.10.11 for environment consistency(both venv and conda , whoever uses whichever)
2.Anything that you install must also be mentioned here so that teammates can install those packages at the time of integration using:
pip install -r requirements.txt
3.Do not change JSON structure
4.Do not push .venv
5.Do not push large experimental files
6.Do not modify files outside your assigned module
7.Do not change the project skeleton (folder structure)
8.Do not create any main file yet as it will be created at the time on integration , test modules independently
9.Use meaningful commit messages
10.Pull latest changes before pushing (git pull origin main)


##any more instructions may be added on under this section


#Project Overview
This project implements an Unsupervised Aspect-Based Sentiment Analysis (ABSA) system.
Dataset structure:
Overall 10 EV Models explored with 15 reviews each which leads to 150 total reviews,done to ensure dataset quality and diversity. 
Reviews collected from:
1.https://www.cardekho.com/tata/nexon-ev/user-reviews/2?subtab=latest

The system processes raw reviews and performs:
1.Text Preprocessing  
2.Aspect Identification
3.Sentiment Analysis per Aspect  

Final Output:  
A dictionary mapping each aspect to its corresponding sentiment.

The project is divided into 3 parts:

Part 1 – Preprocessing
Folder: `part1_preprocessing/`  
File: `preprocess.py`

Input:
- `data/raw_reviews.json`

Output:
- `data/preprocessed_output.json`

Responsibilities:
- Tokenization
- Stopword removal
- Lemmatization
- Clean token list generation


Part 2 – Aspect Identification
Folder: `part2_aspect_identification/`  
File: `aspect_extraction.py`

Input:
- `data/preprocessed_output.json`

Output:
- `data/aspect_output.json`

Responsibilities:
- POS tagging
- Noun extraction
- Identifying aspect terms


Part 3 – Sentiment Analysis
Folder: `part3_sentiment_analysis/`  
File: `sentiment_analysis.py`

Input:
- `data/preprocessed_output.json`
- `data/aspect_output.json`

Output:
- `outputs/final_aspect_sentiment.json`

Responsibilities:
- Extract context window around aspects
- Determine polarity (Positive / Negative / Neutral)
- Generate final {aspect: sentiment} mapping

