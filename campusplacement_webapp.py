import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('E:/python_programs/campusplacement/trained_model.sav','rb'))

def placement(input_data):

    raw_data = np.asarray(input_data)
    if raw_data[0] == 'M':
        raw_data[0] = 0
    else:
        raw_data[0] = 1
    if raw_data[1] == 'A':
        raw_data[1] = 0
    else:
        raw_data[1] = 1
    if raw_data[6] == 'Yes':
        raw_data[6] = 1
    else:
        raw_data[6] = 0
    input_data_reshaped = raw_data.reshape(1, -1)
    # input_data_reshaped=input_data_as_np_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if prediction[0] == 0:
        return 'student will not be placed'
    else:
        return 'student will be placed'

def main():

    st.title('Campus Placement Prediction')


    Gender=st.selectbox('Gender',['M','F'])
    Section = st.selectbox('Section', ['A','B'])
    SSC_Percentage = st.number_input('SSC Percentage',min_value=0.0,max_value=100.0,step=0.1)
    inter_Diploma_percentage = st.number_input('inter_Diploma_percentage', min_value=0.0, max_value=100.0, step=0.1)
    BTech_percentage = st.number_input('B.Tech_percentage', min_value=0.0, max_value=100.0, step=0.1)
    Backlogs = st.number_input('Backlogs', min_value=0, max_value=40, step=1)
    Register= st.selectbox('registered_for_ Placement_Training', ['Yes','No'])

    placement_status=''

    if st.button('Placement Status'):
        placement_status = placement([Gender,Section,SSC_Percentage,inter_Diploma_percentage,BTech_percentage,Backlogs,Register])

    st.success(placement_status)


if __name__=='__main__':
    main()


