# Disaster-Response-Pipelines

## Project Description

This project will analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

The data set contains real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

## Project Components
1. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database
 
2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App
- Modify file paths for database and model as needed
- Add data visualizations using Plotly in the web app. One example is provided for you
![screenshot1](https://github.com/JcFreya/Disaster-Response-Pipelines/blob/master/img/Screen%20Shot%202019-08-15%20at%203.57.54%20PM.png)
![screenshot2](https://github.com/JcFreya/Disaster-Response-Pipelines/blob/master/img/Screen%20Shot%202019-08-15%20at%203.58.22%20PM.png)

### Install

### Code

File structure of the project:

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```

### Run

1. Data Pipelines: Python Scripts

```bash
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```  

2. Running the Web App
  Use terminal commands to navigate inside the folder with the run.py file
```bash
python run.py
```
Open another Terminal Window
```
env|grep WORK
```
In a new web browser window, type in the following:
```
https://SPACEID-3001.SPACEDOMAIN
```

### Data

The data set contains real messages that were sent during disaster events.


