import streamlit as st
import pandas as pd
import joblib

st.header("Parkinson's Disease detection")
st.markdown('***')
FoHz = st.number_input('Average vocal fundamental frequency')
FhiHz = st.number_input('Maximum vocal fundamental frequency')
FloHz = st.number_input('Minimum vocal fundamental frequency')
jitterPer = st.number_input('MDVP:Jitter(%)')
jitterAbs = st.number_input('MDVP:Jitter(Abs)')
rap = st.number_input('MDVP:RAP')
ppq = st.number_input('MDVP:PPQ')
ddp = st.number_input('Jitter:DDP')
shimmer = st.number_input('MDVP:Shimmer')
shimmerDB = st.number_input('MDVP:Shimmer(dB)')
shimmerAPQ3 = st.number_input('Shimmer:APQ3')
shimmerAPQ5 = st.number_input('Shimmer:APQ5')
apq = st.number_input('MDVP:APQ')
shimmerDDA = st.number_input('Shimmer:DDA')
nhr = st.number_input('NHR')
hnr = st.number_input('HNR')
rpde = st.number_input('RPDE')
dfa = st.number_input('DFA')
spread1 = st.number_input(
    'spread1')
spread2 = st.number_input(
    ' spread2')
d2 = st.number_input('D2')
ppe = st.number_input('PPE')


# detect
if st.button('Detect'):
    model = joblib.load(
        '/model/svm.pkl')

    F = pd.DataFrame([[FoHz, FhiHz, FloHz, jitterPer, jitterAbs,
                       rap, ppq, ddp, shimmer, shimmerDB,shimmerAPQ5, shimmerAPQ5, apq,
                       shimmerDDA, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]], columns=['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                                                                                              'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                                                                                              'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
                                                                                              'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'])

    prediction = model.predict(F)[0]

    if prediction == 0:
        st.write("parkinson's disease")
    else:
        st.write("Normal")