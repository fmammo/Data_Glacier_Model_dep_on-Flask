# Import neccessary libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

hire_data = pd.read_csv('hire.csv')
hire_data.head()

y = hire_data['salary'] # Our target value
x = hire_data[['exper','test','interv']] # our feature value 

# Since we have a very small dataset we will train our model with all available data
reg_model = LinearRegression()

# fitting model with the trainning data
reg_model.fit(x,y)

pickle.dump(reg_model, open('model.pickle','wb'))

