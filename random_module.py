import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')


food_data = pd.read_csv('data/food_v1.csv')

X = food_data.drop('Food_items', axis=1)
y = food_data['Food_items']

available_cols = ['Calories', 'Proteins', 'Carbohydrates', 'Fibre']  #0 1 2 4

user_input = pd.DataFrame({
    'nutrient': ['Calories','Proteins', 'Carbohydrates', 'Sugars', 'Fibre', 'Fats','VitaminD', 'Calcium', 'Iron', 'Sodium', 'Potassium'],
    'min_value': [61, 0.14, 14.08, 4.04, 2.0 ,1.28, 0, 13, 0.28, 0, 104],
    'max_value': [61, 0.14, 14.08, 4.04, 2.0 ,1.28, 0, 13, 0.28, 0, 104]
})



def calculate_bmi(height, weight):
    height /= 100
    bmi = weight/np.square(height)
    return round(bmi,2)

def get_min_max(col_name, df):
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3-q1
    q_min = q1-(1.5*iqr)
    q_max = q3+(1.5*iqr)

    return q_min,q1,q3,q_max    

def weight_category(bmi):
    if(bmi<18.5):
        weight = 0
    elif(bmi>=18.5 and bmi<=24.9):
        weight = 1
    else:
        weight = 2

    if(weight == 0):
        for i, col in enumerate(available_cols):
            q_min,q1,q3,q_max = get_min_max(col, food_data)
            if(i == 3):
                user_input['min_value'][i+1] = q3
                user_input['max_value'][i+1] = q_max
            else:
                user_input['min_value'][i] = q3
                user_input['max_value'][i] = q_max
    elif(weight == 1):
        for i, col in enumerate(available_cols):
            q_min,q1,q3,q_max = get_min_max(col, food_data)
            if(i == 3):
                user_input['min_value'][i+1] = q1
                user_input['max_value'][i+1] = q3
            else:
                user_input['min_value'][i] = q1
                user_input['max_value'][i] = q3
    else:
        for i, col in enumerate(available_cols):
            q_min,q1,q3,q_max = get_min_max(col, food_data)
            if(i == 3):
                user_input['min_value'][i+1] = q_min
                user_input['max_value'][i+1] = q1
            else:
                user_input['min_value'][i] = q_min
                user_input['max_value'][i] = q1
                
    return weight


def main_fun(height, weight):
    recommended_items = []
    tree_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    tree_classifier.fit(X, y)
    
    bmi = calculate_bmi(height, weight)
    weight = weight_category(bmi)
    
    all_recommended_items = {}
    
    for i, col in enumerate(available_cols):
        nutrient_range = ['high', 'balanced', 'low']
        
        if(i==3):
            min_value = user_input['min_value'][i+1]
            max_value = user_input['max_value'][i+1]
        else:
            min_value = user_input['min_value'][i]
            max_value = user_input['max_value'][i]
            
        filtered_data = food_data[(food_data[col] >= min_value) & (food_data[col] <= max_value)]
        if(filtered_data.empty):
            continue
        # Predict the meal category based on the user's preferences
        predicted_category = tree_classifier.predict(filtered_data.drop('Food_items', axis=1))
        
        # Get a list of foods in the predicted category
        recommended_foods = filtered_data[filtered_data['Food_items'] == predicted_category[0]]
        
        if(weight == 0):
            food_range = nutrient_range[0]
        elif(weight == 1):
            food_range = nutrient_range[1]
        else:
            food_range = nutrient_range[2]
            
        # print(f"For {nutrient_range[2]} {col} foods, we recommend {predicted_category[0:5]} foods")
        # print('--------------------------------------------------------------------------------')
        
        all_recommended_items[f'{food_range} {col}'] = predicted_category[0:5]
        
    return all_recommended_items
        

# print(main_fun(168,70))

