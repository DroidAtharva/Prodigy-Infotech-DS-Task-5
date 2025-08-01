# Prodigy-Infotech-DS-Task-5
# US Traffic Accident Analysis (Exploratory Data Analysis in Jupyter Notebook)

This Jupyter Notebook project explores a large dataset of US traffic accidents with the goal of identifying patterns, high-risk areas, and visualizing trends over time and geography.

## Description
The notebook walks through:
- Data Cleaning
- Handling Missing Values
- Exploratory Data Analysis (EDA)
- Visualizations using Seaborn & Matplotlib
- Temporal, Spatial, and Severity-based insights

## Dataset
- File: US_Accidents_March23.csv
- Source: Kaggle – US Accidents (Sobhan Moosavi)
- Records: 2.8 million+ accident entries
- Features: Location, time, weather, severity, and more

## Technologies Used
- Python
- Jupyter Notebook
- Pandas
- Matplotlib
- Seaborn

## How to Run
1. Install required packages:
   pip install pandas matplotlib seaborn jupyter

2. Launch the notebook:
   jupyter notebook Task\ 5.ipynb

3. Ensure `US_Accidents_March23.csv` is in the same folder as the notebook.

## Key Analysis Performed

### Data Cleaning
- Dropped irrelevant and highly null columns
- Imputed missing categorical data using mode
- Removed duplicates

### Visualizations
- Top/Bottom cities with accidents
- State-wise accident counts
- Accident severity distribution (Pie Chart)
- Year-wise and Time-of-Day trends
- Weather condition impact
- Geospatial scatter plots

## Insights Discovered
- California and Miami report the most accidents.
- Accidents are most frequent in the morning hours.
- Severity 2 is the most common classification.
- Fair weather is most often associated with accidents.
- Year 2021 saw the highest number of cases.

## Outcome
The notebook provides a deep dive into the accident data, helping uncover:
- Temporal trends
- Geographic hotspots
- Factors that influence accident severity

These insights are valuable for traffic policy planning, public safety efforts, and building smarter urban mobility systems.

## Learnings
- Handling real-world messy data
- Visual storytelling with charts
- Extracting patterns from temporal and spatial datasets
- Building actionable insights for real-life applications


CSV File-https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
