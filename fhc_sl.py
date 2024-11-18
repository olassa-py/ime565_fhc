# Import necessary libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Fetal Health Classification: A Machine Learning App')
st.image('fetal_health_image.gif', use_column_width = True) 
st.write("Utilize our Machine Learning application to predict fetal health classifications.")

# Load the pre-trained model from the pickle file
dt_pickle = open('dt_fhc.pickle', 'rb') 
dt_clf = pickle.load(dt_pickle) 
dt_pickle.close()

rf_pickle = open('rf_fhc.pickle', 'rb') 
rf_clf = pickle.load(rf_pickle) 
rf_pickle.close()

ab_pickle = open('ab_fhc.pickle', 'rb') 
ab_clf = pickle.load(ab_pickle) 
ab_pickle.close()

sv_pickle = open('sv_fhc.pickle', 'rb') 
sv_clf = pickle.load(sv_pickle) 
sv_pickle.close()

# Create a sidebar for input collection
st.sidebar.header('Fetal Health Features Input')

# Load the default dataset
base_df = pd.read_csv('fetal_health.csv')
sampledf = base_df.head().drop(columns = 'fetal_health')

# File Upload
file_upload = st.sidebar.file_uploader("Drag and drop file here", type = "csv", help = "File must be in CSV format")
st.sidebar.warning("⚠️&nbsp;&nbsp; Ensure your data strictly follows the format outlined below")
st.sidebar.write(sampledf)

# Sidebar input for model using
decision = st.sidebar.radio('Choose Model for Prediction', options = ['Random Forest', 'Decision Tree', 'AdaBoost','Soft Voting'])
st.sidebar.info(f"You selected: {decision}")

if decision == 'Decision Tree':
    mod = dt_clf
    modt = 'Decision Tree'
    mod_fi = 'fhc_dt_fi.svg'
    mod_cm = 'fhc_dt_cm.svg'
    mod_cr = 'fhc_dt_cr.csv'
elif decision == 'Random Forest':
    mod = rf_clf
    modt = 'Random Forest'
    mod_fi = 'fhc_rf_fi.svg'
    mod_cm = 'fhc_rf_cm.svg'
    mod_cr = 'fhc_rf_cr.csv'
elif decision == 'AdaBoost':
    mod = ab_clf
    modt = 'AdaBoost'
    mod_fi = 'fhc_ab_fi.svg'
    mod_cm = 'fhc_ab_cm.svg'
    mod_cr = 'fhc_ab_cr.csv'
elif decision == 'Soft Voting':
    mod = sv_clf
    modt = 'Soft Voting'
    mod_fi = 'fhc_sv_fi.svg'
    mod_cm = 'fhc_sv_cm.svg'
    mod_cr = 'fhc_sv_cr.csv'

if file_upload is None:
    ### I used ChatGPT to get a space between the emoji and text without breaking the italices
    st.info('ℹ️&nbsp;&nbsp; *Please upload data to proceed.*')
else:
    st.success("✅&nbsp;&nbsp; *CSV file uploaded successfully.*")
    uploaded_data = pd.read_csv(file_upload)
    new_data = uploaded_data.copy()

    st.subheader(f'Predicting Fetal Health Class using {modt} Model')

   # Add missing columns if needed and reorder columns
    missing_cols = [col for col in base_df.columns if col not in new_data.columns]
    for col in missing_cols:
        new_data[col] = 0
    
    # Set order for the df
    order = [
    'baseline value',
    'accelerations',
    'fetal_movement',
    'uterine_contractions',
    'light_decelerations',
    'severe_decelerations',
    'prolongued_decelerations',
    'abnormal_short_term_variability',
    'mean_value_of_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability',
    'mean_value_of_long_term_variability',
    'histogram_width',
    'histogram_min',
    'histogram_max',
    'histogram_number_of_peaks',
    'histogram_number_of_zeroes',
    'histogram_mode',
    'histogram_mean',
    'histogram_median',
    'histogram_variance',
    'histogram_tendency']
    new_data = new_data[order]

    # Make Prediction and probabilities
    prediction = mod.predict(new_data)
    prediction_probs = mod.predict_proba(new_data).max(axis=1) * 100

    # Add predicted class to df
    ### Used ChatGPT to get the <.replace({1: 'Normal', 2: 'Suspect', 3: 'Pathological'})> since I couldn't figure out how to replace them within the dataframe
    new_data['Predicted Class'] = prediction
    new_data = new_data.sort_values(by = 'Predicted Class', ascending = True)
    new_data['Predicted Class'] = new_data['Predicted Class'].replace({1: 'Normal', 2: 'Suspect', 3: 'Pathological'})
    
    # Add prediction probabilities to df
    ### Used ChatGPT to remove the excess 0s with the code <.apply(lambda x: f'{x:.1f}')>
    new_data['Prediction Probability'] = prediction_probs.round(1) #.apply(lambda x: f'{x:.1f}')

    # Function to apply cell color based on prediction class
    ### Used ChatGPT to get the 'background-color: lime' and other lines for distinguishing classes
    def highlight_class(val):
        color = ''
        if val == 'Normal':
            color = 'background-color: lime'
        elif val == 'Suspect':
            color = 'background-color: yellow'
        elif val == 'Pathological':
            color = 'background-color: orange'
        return color

    # Display the styled DataFrame
    ### Used ChatGPT to apply the class
    st.dataframe(new_data.style.applymap(highlight_class, subset=['Predicted Class']))

    #----------------------------------------------------------
    # Showing additional items in tabs
    st.subheader("Model Performance and Insights")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])
   
    # Tab 1: Confusion Matrix
    with tab1:
        st.write("### Confusion Matrix")
        st.image(mod_cm)
        st.caption("Confusion Matrix of model predictions.")

    # Tab 2: Classification Report
    with tab2:
        st.write("### Classification Report")
        report_df = pd.read_csv(mod_cr, index_col = 0).transpose()
        col = 'binary'
        if decision == 'Decision Tree':
            col = 'Greens'
        elif decision == 'Random Forest':
            col = 'Greens'
        elif decision == 'AdaBoost':
            col = 'Reds'
        elif decision == 'Soft Voting':
            col = 'PuOr'
        st.dataframe(report_df.style.background_gradient(cmap= col).format(precision=2))
        st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")
    
    # Tab 3: Feature Importance Visualization
    with tab3:
        st.write("### Feature Importance")
        st.image(mod_fi)
        st.caption("Features used in this prediction are ranked by relative importance.")

