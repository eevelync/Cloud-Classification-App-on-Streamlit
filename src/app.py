import os
import boto3
import pickle
import streamlit as st
import pandas as pd
import warnings
import logging
from botocore.exceptions import BotoCoreError
from sklearn.model_selection import train_test_split
from typing import Tuple
import numpy as np

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

@st.cache_resource(ttl=1 * 24 * 60 * 60)
def load_data() -> Tuple:
    """
    Load data from S3 bucket.

    Returns:
        Tuple: A tuple containing the loaded train data DataFrames.
    """
    try:
        s3 = boto3.client('s3')

        # Load train data from S3
        train_data = s3.get_object(Bucket='scn3674-test-0', Key='experiments/train.csv')
        train_df = pd.read_csv(train_data['Body'], index_col=None)
        train2_data = s3.get_object(Bucket='scn3674-test-0', Key='experiments_v2/train.csv')
        train2_df = pd.read_csv(train2_data['Body'], index_col=None)
    except BotoCoreError as e:
        logging.error('Error loading data from S3', exc_info=True)
        st.error('Error loading data. Please check the logs for more details.')
        return None, None
    except Exception as e:
        logging.error('Unexpected error', exc_info=True)
        st.error('Unexpected error. Please check the logs for more details.')
        return None, None

    return train_df, train2_df

@st.cache_resource(ttl=1 * 24 * 60 * 60)
def load_models() -> Tuple:
    """
    Load models from S3 bucket.

    Returns:
        Tuple: A tuple containing the loaded model objects.
    """
    try:
        # Load model version 1
        s3 = boto3.client('s3')
        bucket_name = 'scn3674-test-0'
        object_name = 'experiments/trained_model_object.pkl'
        file_path = 'trained_model_object.pkl'
        s3.download_file(bucket_name, object_name, file_path)
        with open(file_path, 'rb') as f:
            model1 = pickle.load(f)

        # Load model version 2
        bucket_name = 'scn3674-test-0'
        object_name = 'experiments_v2/trained_model_object.pkl'
        file_path = 'trained_model_object.pkl'
        s3.download_file(bucket_name, object_name, file_path)
        with open(file_path, 'rb') as f:
            model2 = pickle.load(f)
    except BotoCoreError as e:
        logging.error('Error loading models from S3', exc_info=True)
        st.error('Error loading models. Please check the logs for more details.')
        return None, None
    except Exception as e:
        logging.error('Unexpected error', exc_info=True)
        st.error('Unexpected error. Please check the logs for more details.')
        return None, None

    return model1, model2

def process_data_and_predict(model, df, features):
    """
    Process data and make predictions using the provided model.

    Args:
        model: The model object.
        df (pd.DataFrame): The data DataFrame.
        features (list): The list of features to use.

    Returns:
        Tuple: A tuple containing the prediction DataFrame and the predicted probabilities DataFrame.
    """
    try:
        df = [features]
        # Make predictions and convert to DataFrame
        predictions = model.predict(df)
        predictions_df = pd.DataFrame(predictions, columns=['Predicted Class'])

        # Generate predicted probabilities and convert to DataFrame
        probabilities = model.predict_proba(df)
        df_probabilities = pd.DataFrame(probabilities, columns=['Prob Class0', 'Prob Class1'])
        # Round probabilities to 2 decimal places
        df_probabilities = df_probabilities.applymap(lambda x: f'{x:.2f}')

        return predictions_df, df_probabilities
    except Exception as e:
        logging.error('Unexpected error during prediction', exc_info=True)
        st.error('Unexpected error during prediction. Please check the logs for more details.')
        return pd.DataFrame(), pd.DataFrame()

# Streamlit app
st.title('Cloud Type Classification Model')
option = st.sidebar.selectbox("Select Model Version", ('Model 1', 'Model 2'))

if option == 'Model 1':
    # Load train df from S3
    train_df, _ = load_data()
    if train_df is None:
        st.stop()

    # Load model1 from S3
    model1, _ = load_models()
    if model1 is None:
        st.stop()

    # Get model parameters and convert to DataFrame
    parameters1 = model1.get_params()
    df_parameters1 = pd.DataFrame(list(parameters1.items()), columns=['Parameter', 'Value'])
    df_parameters1['Value'] = df_parameters1['Value'].astype(str)

    st.write("Below are the model parameters for model 1")
    st.dataframe(df_parameters1, width=400)

    # Keep features used for model training
    st.sidebar.header("Input Parameters")
    log_entropy = st.sidebar.slider('log_entropy',-4,0,1)
    IR_norm_range = st.sidebar.slider('IR_norm_range', 0,80,6)
    entropy_x_contrast = st.sidebar.slider('entropy_x_contrast', 0,165,12)

    features = [log_entropy, IR_norm_range, entropy_x_contrast]

    # Process data and predict
    predictions1_df, df_probabilities1 = process_data_and_predict(model1, train_df, features)

    st.write("Below is the prediction from model 1")
    st.dataframe(predictions1_df, width=400)
    st.write("Below are the predicted probabilities from model 1")
    st.dataframe(df_probabilities1, width=800)

elif option == 'Model 2':
    # Load train df from S3
    _, train2_df = load_data()
    if train2_df is None:
        st.stop()

    # Load model2 from S3
    _, model2 = load_models()
    if model2 is None:
        st.stop()

    # Get model parameters and convert to DataFrame
    parameters2 = model2.get_params()
    df_parameters2 = pd.DataFrame(list(parameters2.items()), columns=['Parameter', 'Value'])
    df_parameters2['Value'] = df_parameters2['Value'].astype(str)

    st.write("Below are the model parameters for model 2")
    st.dataframe(df_parameters2, width=400)

    # Keep features used for model training
    st.sidebar.header("Input Parameters")
    log_entropy = st.sidebar.slider('log_entropy', -4,0,-2)
    IR_norm_range = st.sidebar.slider('IR_norm_range', 0,80,6)
    entropy_x_contrast = st.sidebar.slider('entropy_x_contrast', 0,165,12)

    features = [log_entropy, IR_norm_range, entropy_x_contrast]

    # Process data and predict
    predictions2_df, df_probabilities2 = process_data_and_predict(model2, train2_df, features)

    st.write("Below is the prediction from model 2")
    st.dataframe(predictions2_df, width=400)
    st.write("Below are the predicted probabilities from model 2")
    st.dataframe(df_probabilities2, width=800)