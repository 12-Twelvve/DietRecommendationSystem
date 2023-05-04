import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv('data/food.csv')
Food_itemsdata=data['Food_items']

def calBmi(weight, height):
    bmi = weight/((height/100)**2) 
    print("Your body mass index is: ", round(bmi,2))
    if ( bmi < 16):
        val = "Acoording to your BMI, you are Severely Underweight"
        bfbmi=2
        lbmi=2
        dbmi=3
    elif ( bmi >= 16 and bmi < 18.5):
        val = "Acoording to your BMI, you are Underweight"
        bfbmi=4
        lbmi=0
        dbmi=2
    elif ( bmi >= 18.5 and bmi < 25):
        val = "Acoording to your BMI, you are Healthy"
        bfbmi=3
        lbmi=3
        dbmi=0
    elif ( bmi >= 25 and bmi < 30):
        val = "Acoording to your BMI, you are Overweight"
        bfbmi=1
        lbmi=4
        dbmi=4
    elif ( bmi >=30):
        val = "Acoording to your BMI, you are Severely Overweight"
        bfbmi=0
        lbmi=1
        dbmi=1
    return bfbmi, lbmi, dbmi,val

def clusterKmeans(catdata):
    ## K-Means
    X = np.array(catdata[:,1:len(catdata)])
    kmeans = KMeans(n_clusters=5, init='k-means++', 
                    max_iter=300, tol=0.0001, algorithm='elkan', 
                    random_state=0).fit(X)
    lbl=kmeans.labels_
    return lbl

def breakFast():
    Breakfastdata=data['Breakfast']
    BreakfastdataNumpy=Breakfastdata.to_numpy()
    breakfastfoodseparated=[] #name of breakfast foods
    breakfastfoodseparatedID=[] #id of breakfast foods
    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i]==1:
            breakfastfoodseparated.append( Food_itemsdata[i] )
            breakfastfoodseparatedID.append(i)
   
    # retrieving Breafast data rows by loc method 
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    val=list(np.arange(5,16))
    Valapnd=[0]+val
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    # cluster
    brklbl = clusterKmeans(breakfastfoodseparatedIDdata.to_numpy())
    bf_df = breakfastfoodseparatedIDdata
    bf_df['label']  = brklbl
    # saving the dataframe
    bf_df.to_csv('data/breakfast.csv')
    

def lunch():
    Lunchdata=data['Lunch']
    LunchdataNumpy=Lunchdata.to_numpy()
    lunchfoodseparated=[] #name of breakfast foods
    lunchfoodseparatedID=[] #id of breakfast foods
    for i in range(len(Lunchdata)):
        if LunchdataNumpy[i]==1:
            lunchfoodseparated.append( Food_itemsdata[i] )
            lunchfoodseparatedID.append(i)

    # retrieving Breafast data rows by loc method 
    lunchfoodseparatedIDdata = data.iloc[lunchfoodseparatedID]
    lunchfoodseparatedIDdata=lunchfoodseparatedIDdata.T
    val=list(np.arange(5,16))
    Valapnd=[0]+val
    lunchfoodseparatedIDdata=lunchfoodseparatedIDdata.iloc[Valapnd]
    lunchfoodseparatedIDdata=lunchfoodseparatedIDdata.T

    # cluster
    lunchlbl = clusterKmeans(lunchfoodseparatedIDdata.to_numpy())
    l_df = lunchfoodseparatedIDdata
    l_df['label']  = lunchlbl
    # saving the dataframe
    l_df.to_csv('data/lunch.csv')

def dinner():
    Dinnerdata=data['Dinner']
    DinnerdataNumpy=Dinnerdata.to_numpy()
    dinnerfoodseparated=[] #name of breakfast foods
    dinnerfoodseparatedID=[] #id of breakfast foods
    for i in range(len(Dinnerdata)):
        if DinnerdataNumpy[i]==1:
            dinnerfoodseparated.append( Food_itemsdata[i] )
            dinnerfoodseparatedID.append(i)

    # retrieving Breafast data rows by loc method 
    dinnerfoodseparatedIDdata = data.iloc[dinnerfoodseparatedID]
    dinnerfoodseparatedIDdata=dinnerfoodseparatedIDdata.T
    val=list(np.arange(5,16))
    Valapnd=[0]+val
    dinnerfoodseparatedIDdata=dinnerfoodseparatedIDdata.iloc[Valapnd]
    dinnerfoodseparatedIDdata=dinnerfoodseparatedIDdata.T

    # cluster
    dinnerlbl = clusterKmeans(dinnerfoodseparatedIDdata.to_numpy())
    d_df = dinnerfoodseparatedIDdata
    d_df['label']  = dinnerlbl
    # saving the dataframe
    d_df.to_csv('data/dinner.csv')
    

def makeTrainingset(cat, lbl, bmicls, agecls):
    ## train set
    catfin=np.zeros((len(cat)*5,6),dtype=np.float32)
    t = 0
    yt = []
    for zz in range(5):
        for jj in range(len(cat)):
            valloc=list(cat[jj]) #weightloss major columns -4
            valloc.append(bmicls[zz]) #add bmi value randomly
            valloc.append(agecls[zz]) #add age value  randomly
            catfin[t]=np.array(valloc)
            yt.append(lbl[jj]) #randomly assigned the label for breakfast clusters
            t+=1
    return catfin, yt

# ----------------------  xxxxx
def testDataGeneration(cat):
    #test data generation 
    X_test=np.zeros((len(cat),6),dtype=np.float32)
    # age weight height veg
    agecl, clbmi = calBmi(45.0,70.0,169.0)
    ti=(clbmi+agecl)/2
    for jj in range(len(cat)):
        valloc=list(cat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj]=np.array(valloc)*ti
    return X_test

# ----------------------  xxxxx
def forestClassifier(X_train, y_train):
    clf=RandomForestClassifier(n_estimators=100).fit(X_train,y_train)
    return clf

def train_model():
    breakFast()
    lunch()
    dinner()
# train_model()

def recommend(w, h):
# --------------
    print(" \n Weight : %s \n Height : %s" % (w, h))
    bf_df = pd.read_csv('data/breakfast.csv')
    l_df = pd.read_csv('data/lunch.csv')
    d_df = pd.read_csv('data/dinner.csv')
    bfbmi, lbmi, dbmi,val = calBmi(int(w), int(h))
    
    # print ('\n SUGGESTED BREAKFAST ITEMS :: \n')
    breakfast_list = (bf_df.loc[bf_df['label']==bfbmi]['Food_items']).to_string(index= False)
    breakfast_list = [item.strip() for item in breakfast_list.split('\n')]    
    # print(breakfast_list)
    # print ('\n SUGGESTED LUNCH ITEMS :: \n')
    lunch_list = (l_df.loc[l_df['label']==lbmi]['Food_items']).to_string(index= False)
    lunch_list = [item.strip() for item in lunch_list.split('\n')]
    # print(lunch_list)
    # print ('\n SUGGESTED DINNER ITEMS :: \n')
    dinner_list = (d_df.loc[d_df['label']==dbmi]['Food_items']).to_string(index= False)
    dinner_list=[item.strip() for item in dinner_list.split('\n')]
    # print(dinner_list)
    return breakfast_list, lunch_list, dinner_list, val

# weight = 80
# height = 169
# recommend(weight, height)