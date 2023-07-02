import numpy as np
import pickle

loaded_model=pickle.load(open('E:/python_programs/campusplacement/trained_model.sav','rb'))

input_data=('M','A',85,75,70,2,'Yes')
raw_data=np.asarray(input_data)
if raw_data[0]=='M':
  raw_data[0]=0
else:
  raw_data[0]=1
if raw_data[1]=='A':
  raw_data[1]=0
else:
  raw_data[1]=1
if raw_data[6]=='Yes':
  raw_data[6]=1
else:
  raw_data[6]=0
input_data_reshaped=raw_data.reshape(1,-1)
#input_data_reshaped=input_data_as_np_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if prediction[0]==0:
  print('student will not be placed')
else:
  print('student will be placed')