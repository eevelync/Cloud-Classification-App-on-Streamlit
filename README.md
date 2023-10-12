# Cloud Classification App

## Overview
The Cloud Classification App is a Streamlit web application that utilizes machine learning for cloud classification. It hosts two versions of a trained machine learning model, giving users the flexibility to select the model version of their choice and to interactively input data via the sidebar for real-time predictions.

## Setup and Usage
To use this application, you will need to set up a Python environment with the necessary dependencies. It's recommended that this is done within a virtual environment to avoid any potential conflicts with your system's Python or other Python applications you might be using.

Here are the steps to get the application up and running:

1. Clone the repository:
```
git clone https://github.com/MSIA/423-2023-hw3-scn3674.git
```

2. Create a virtual environment:
```
python3 -m venv .venv 
```

3. Activate the virtual environment
```
source .venv/bin/activate
```

4. Install the required Python packages:
```
pip install -r requirements.txt
```

5. Run the Streamlit app:
```
streamlit run src/app.py
```

## Model Selection
To select a model, use the **dropdown menu** on the left sidebar of the Streamlit app. Here you can select from two versions of the trained model with different parameters. You can also input various parameters through the interactive features available on the sidebar. As you change the input values, the predictions will update in real-time.

## Docker
If you prefer to run the application in a Docker container, use the following commands:

1. Build the Docker image:
```
docker build -t cloudapp .
```

2. Run the Docker container

```
docker run -p 80:80 cloudapp  
```

## Deployment

This application is deployed on AWS ECS, which can access the live Streamlit application using the following link:

[Cloud Classification App](http://cloudapp-827189284.us-east-2.elb.amazonaws.com/)






