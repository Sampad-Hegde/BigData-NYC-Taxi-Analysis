from pyspark import SparkContext,SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
import seaborn as sns
import matplotlib.dates as dates
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.metrics import accuracy_score


sc = SparkContext()
sqlContext = SQLContext(sc)

def taxi_data_cleaner_pandas(data):
    
    data.columns = map(str.lower, data.columns)
    
    # Getting rid of rides with 0 passengers
    data = data.loc[data.passenger_count > 0]
    
    # Getting rid of rides with impossibly low fares
    data = data.loc[data.fare_amount > 1.]
    
    #Making a feature which measures the % a driver was tipped 
    data["tip_percentage"] =  100.*(data["tip_amount"])/(data["total_amount"] - data["tip_amount"])

    
    # Adding columns that give the month , day of the week and hour of pickups and dropoffs
    
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])
    
    data['pickup_hour'] = data['pickup_datetime'].dt.hour
    data['dropoff_hour'] = data['dropoff_datetime'].dt.hour
    data['pickup_month'] = data['pickup_datetime'].dt.month
    data['dropoff_month'] = data['dropoff_datetime'].dt.month
    data['pickup_dayofweek'] = data['pickup_datetime'].dt.dayofweek
    data['dropoff_dayofweek'] = data['dropoff_datetime'].dt.dayofweek
    
    # print("P.H : ",data['pickup_hour'],"D.H : ",data['dropoff_hour'],"D.M : ",data['dropoff_month'],"D.D.W : ",data['dropoff_dayofweek'])

    return(data)


# This function
def feature_mapper(feature, feature_map_values):
    """
    feature: feature in question
    feature_map_values: values that the continuous values in 'feature' will be mapped to"
    """
    final = []
    for i in feature:
        i_use = feature_map_values[np.abs(feature_map_values  - i) == np.min(np.abs(feature_map_values  - i))][0]
        final.append(i_use)
    return(final)

green_2014 = sqlContext.read.csv("hdfs://localhost:9000/Dataset/Green_2014.csv", header=True,inferSchema=True)
yellow_2014 = sqlContext.read.csv("hdfs://localhost:9000/Dataset/Yellow_2014.csv", header=True,inferSchema=True)

green_2014.printSchema()
yellow_2014.printSchema()

green_2014_credit = green_2014.filter(green_2014.Payment_type == 1)
yellow_2014_credit = yellow_2014.filter(yellow_2014.payment_type == 'CRD')

green_2014_subsample = green_2014_credit.sample(False, 0.1, seed=0)
yellow_2014_subsample = yellow_2014_credit.sample(False, 0.01, seed=0)

green_2014_subsample_use = green_2014_subsample.toPandas()
yellow_2014_subsample_use = yellow_2014_subsample.toPandas()

green_2014_subsample_use = green_2014_subsample_use.sample(n = 100000)
yellow_2014_subsample_use = yellow_2014_subsample_use.sample(n = 100000)

green_2014_subsample_use = taxi_data_cleaner_pandas(green_2014_subsample_use)
yellow_2014_subsample_use =taxi_data_cleaner_pandas(yellow_2014_subsample_use)

target_green = green_2014_subsample_use["tip_amount"]
target_yellow = yellow_2014_subsample_use["tip_amount"]


columns = ['pickup_dayofweek', 'pickup_month', 
           'dropoff_dayofweek', 'dropoff_month', 'dropoff_hour', 
           'tolls_amount', 
           # 'pickup_neighborhood', 'dropoff_neighborhood', 
           'passenger_count'
           ]


green_2014_subsample_use= green_2014_subsample_use[columns]
yellow_2014_subsample_use  = yellow_2014_subsample_use[columns]

"""
neighborhoods_list = set(list(yellow_2014_subsample_use.dropoff_neighborhood.unique()) +
                         list(yellow_2014_subsample_use.pickup_neighborhood.unique())  + 
                          list(green_2014_subsample_use.pickup_neighborhood.unique())  +
                          list(green_2014_subsample_use.dropoff_neighborhood.unique()))

neighborhoods_number_list = np.arange(len(neighborhoods_list))
neighborhood_mapper = dict(zip(neighborhoods_list,neighborhoods_number_list  ))

yellow_2014_subsample_use_done = yellow_2014_subsample_use.replace({"pickup_neighborhood": neighborhood_mapper})
yellow_2014_subsample_use_done = yellow_2014_subsample_use_done.replace({"dropoff_neighborhood": neighborhood_mapper})


green_2014_subsample_use_done = green_2014_subsample_use.replace({"pickup_neighborhood": neighborhood_mapper})
green_2014_subsample_use_done = green_2014_subsample_use_done.replace({"dropoff_neighborhood": neighborhood_mapper})"""



target_green_binned  = feature_mapper(np.array(target_green), np.arange(0, 12., 2))
target_yellow_binned  = feature_mapper(np.array(target_yellow), np.arange(0, 12., 2))

#MODEL BUILDING

X_green = green_2014_subsample_use
Y_green = target_green_binned 
X_train_green, X_test_green,Y_train_green, Y_test_green = train_test_split(X_green,Y_green,test_size = 0.3, random_state = 30)

X_yellow = yellow_2014_subsample_use
Y_yellow = target_yellow_binned 
X_train_yellow, X_test_yellow,Y_train_yellow, Y_test_yellow = train_test_split(X_yellow,Y_yellow,test_size = 0.3, random_state = 30)


model = XGBClassifier()
model.fit(X_train_green, Y_train_green)  
y_pred_green = model.predict(X_test_green)

feature_importances_green = pd.DataFrame(model.feature_importances_)
feature_importances_green['feature'] = columns
feature_importances_green['importance'] = feature_importances_green[0]
feature_importances_green = feature_importances_green.drop(0, axis = 1)


model = XGBClassifier()
model.fit(X_train_yellow, Y_train_yellow)  
y_pred_yellow = model.predict(X_test_yellow)

feature_importances_yellow = pd.DataFrame(model.feature_importances_)
feature_importances_yellow['feature'] = columns
feature_importances_yellow['importance'] = feature_importances_yellow[0]
feature_importances_yellow = feature_importances_yellow.drop(0, axis = 1)


#Graphs


font_axis = 18
plt.style.use('fivethirtyeight')

plt.figure(figsize = (7,12))
plt.suptitle('Feature Importance for Determining Tip Amount', fontsize = 20)
plt.subplots_adjust(left = 0.35, right = 0.9,hspace=0.3, wspace=0.5, top = 0.91, bottom = 0.1)

ax1 = plt.subplot(211)
plt.title('Yellow Taxis')
feature_importances_yellow.sort_values(by = 'importance', ascending = False)[::-1].plot(ax = ax1, kind = 'barh', color = 'gold', legend=None)
plt.ylabel('')
plt.xlabel('Feature Importance', fontsize = font_axis)
plt.yticks(np.arange(len(feature_importances_yellow)), list(feature_importances_yellow.sort_values(by = 'importance', ascending = False)['feature'])[::-1]);



ax2 = plt.subplot(212)
plt.title('Green Taxis')
feature_importances_green.sort_values(by = 'importance', ascending = False)[::-1].plot(ax = ax2, kind = 'barh', color = 'darkgreen', legend=None)
plt.ylabel('')
plt.xlabel('Feature Importance', fontsize = font_axis)
plt.yticks(np.arange(len(feature_importances_green)), list(feature_importances_green.sort_values(by = 'importance', ascending = False)['feature'])[::-1]);

plt.savefig('Results.png', dpi = 1000)


tip_max = 5.

x_yellow = Y_test_yellow - y_pred_yellow
x_green = Y_test_green - y_pred_green

plt.figure(figsize = (5, 9))
plt.suptitle('Error in Taxi Trip Tip Amount Predictions')
plt.subplots_adjust(left = 0.15, hspace=0.45, wspace=0.4, top = 0.88)

ax1 = plt.subplot(211)
plt.title('Yellow Taxis')
sns.kdeplot(x_yellow, color = 'gold', shade=True)
plt.xlim(-tip_max, tip_max)
plt.xlabel(r'$y - y_{pred}$ (dollars) ')
plt.ylabel('Frequency')

ax2 = plt.subplot(212)
plt.title('Green Taxis')
sns.kdeplot(x_green, color = 'darkgreen', shade=True)
plt.xlim(-tip_max, tip_max)
plt.xlabel(r'$y - y_{pred}$ (dollars) ')
plt.ylabel('Frequency')


plt.savefig('Error.png', dpi = 1000)

print (np.std(x_yellow))
print (np.std(x_green))
