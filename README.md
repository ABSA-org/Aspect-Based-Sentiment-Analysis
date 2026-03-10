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
9.Use meaningful commit messages
10.Pull latest changes before pushing (git pull origin main)


##any more instructions may be added on under this section


#Project Overview
This project implements an Unsupervised Aspect-Based Sentiment Analysis (ABSA) system.
Dataset structure:
The study focuses on a single EV model and analyzes approximately 150–200 user reviews. This approach ensures a sufficiently large dataset for meaningful sentiment aggregation while maintaining consistency across product features being evaluated.
Reviews collected from:
https://www.cardekho.com/tata/nexon-ev/user-reviews/2?subtab=latest

The system processes raw reviews and performs:
1.Text Preprocessing  
2.Aspect Identification
3.Sentiment Analysis per Aspect 
4.Aspect Sentiment Aggregation to generate the overall sentiment for each identified aspect.

Final Output:
A dictionary mapping each identified aspect to its aggregated sentiment
distribution and overall sentiment across all reviews.

