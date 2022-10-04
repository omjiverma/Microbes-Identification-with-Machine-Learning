import streamlit as st
import pandas as pd
import time
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Loading the Model Weights
model = joblib.load('xgb_clf.joblib')
classifier = model['xgb_clf']
le = model['label_encoder']

# Function for Converting inputs to a DataFrame
def df_maker(ConvexArea,raddi,FilledArea, MinorAxisLength,Perimeter, Solidity):
    df = pd.DataFrame({'ConvexArea':[ConvexArea], 'raddi':[raddi], 'FilledArea':[FilledArea], 
                       'MinorAxisLength':[MinorAxisLength],'Perimeter':[Perimeter], 'Solidity':[Solidity]})
    return df
 
# App Title And Header           
with st.container():
    st.markdown('<h1><img src="https://cdn-icons-png.flaticon.com/512/2927/2927484.png" width="50">    Microbes Identification App</h1>', unsafe_allow_html=True)
    st.write("#### `Rapid Identification of the Microbes without Genome Sequencing`")
  
#Input Section  
with st.container():    
    col1, col2, col3 = st.columns(3)
    with col1:
        convex_area = st.number_input('Convex Area' ,format='%0.7f')
        raddi = st.number_input('Raddi',format='%0.7f')

    with col2:
        filled_area = st.number_input('Filled Area',format='%0.7f')
        minor_axis_length = st.number_input('Minor Axis Length',format='%0.7f')

    with col3:
        perimeter = st.number_input('Perimeter',format='%0.7f')
        solidity = st.number_input('Solidity',format='%0.7f')



# Button Click Action
if st.button("Identify"):
    st.header('Results')
    # Exception if All inputs are Zero
    if (convex_area == 0.00) and (raddi == 0.00) and (filled_area == 0.00) and (minor_axis_length == 0.00) and (perimeter == 0.00) and (solidity == 0.00):
        st.error('All inputs cannot be zero !',icon="🚨")
        
    else:
        inputs = df_maker(convex_area,raddi,filled_area,minor_axis_length,perimeter,solidity)
        # Prectinting label and probability
        with st.snow():
            pred = classifier.predict(inputs)
            pred_class = (le.inverse_transform(pred))[0]
            pred_prob = classifier.predict_proba(inputs)
            confidence = pred_prob[0][pred][0]
         
# Prediction Result Section
        colnew1, colnew2 = st.columns(2)
        with colnew1:
            st.markdown('### Identified Microbe')
            st.error(pred_class, icon=None)
        with colnew2:
            st.markdown('### Confidence Level')
            st.warning(confidence)
        st.success('Succesfully Identified !')
    
    

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by Omji <a href="https://github.com/omjiverma/Microbes-Identification-with-Machine-Learning/blob/main/test_data.csv">Get Test Data</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
