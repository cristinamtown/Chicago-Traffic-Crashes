## Business Case

Using the City of Chicago Traffic Crashes dataframe, we are going to model the data to try to answer the following questions:

- Primary causes of car crashes in the Chicago Area
- Main types of car crashes
- What can drivers do to prevent crashes?

Focus in this analysis:
- Use a model to predict whether the main cause of crash was driver’s failure to yield


## Methodology:
1. Obtain and merge the traffic crash data from the Chicago Data Portal regarding the crashes and the people 
2. Examine and Scrubbing the data to prepare it for modeling
3. Exploration of the data
4. Perform and evaluate a baseline model of the data
5. Perform and evaluate other models of the data and decide which to use
6. Interpretations and recommendations based on the models and examination of the data.
7. Future work 


## Obtain the Data

Importing, merging, and initial look at the data


```python
# Import pandas and data set
import pandas as pd
crash_df = pd.read_csv('Traffic_Crashes_-_Crashes.csv')
people_df = pd.read_csv('Traffic_Crashes_-_People.csv')
```


```python
# merge the two datasets into one
crash_df = pd.merge(crash_df, people_df, on='CRASH_RECORD_ID', how='left')
```


```python
# preview, make sure everything loaded and merge correctly
crash_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRASH_RECORD_ID</th>
      <th>RD_NO_x</th>
      <th>CRASH_DATE_EST_I</th>
      <th>CRASH_DATE_x</th>
      <th>POSTED_SPEED_LIMIT</th>
      <th>TRAFFIC_CONTROL_DEVICE</th>
      <th>DEVICE_CONDITION</th>
      <th>WEATHER_CONDITION</th>
      <th>LIGHTING_CONDITION</th>
      <th>FIRST_CRASH_TYPE</th>
      <th>...</th>
      <th>EMS_RUN_NO</th>
      <th>DRIVER_ACTION</th>
      <th>DRIVER_VISION</th>
      <th>PHYSICAL_CONDITION</th>
      <th>PEDPEDAL_ACTION</th>
      <th>PEDPEDAL_VISIBILITY</th>
      <th>PEDPEDAL_LOCATION</th>
      <th>BAC_RESULT</th>
      <th>BAC_RESULT VALUE</th>
      <th>CELL_PHONE_USE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>073682ef84ff827659552d4254ad1b98bfec24935cc9cc...</td>
      <td>JB460108</td>
      <td>NaN</td>
      <td>10/02/2018 06:30:00 PM</td>
      <td>10</td>
      <td>NO CONTROLS</td>
      <td>NO CONTROLS</td>
      <td>CLEAR</td>
      <td>DARKNESS</td>
      <td>PARKED MOTOR VEHICLE</td>
      <td>...</td>
      <td>NaN</td>
      <td>NONE</td>
      <td>NOT OBSCURED</td>
      <td>NORMAL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TEST NOT OFFERED</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1560fb8a1e32b528fef8bfd677d2b3fc5ab37278b157fa...</td>
      <td>JC325941</td>
      <td>NaN</td>
      <td>06/27/2019 04:00:00 PM</td>
      <td>45</td>
      <td>NO CONTROLS</td>
      <td>NO CONTROLS</td>
      <td>CLEAR</td>
      <td>DAYLIGHT</td>
      <td>SIDESWIPE SAME DIRECTION</td>
      <td>...</td>
      <td>NaN</td>
      <td>UNKNOWN</td>
      <td>UNKNOWN</td>
      <td>NORMAL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TEST NOT OFFERED</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1560fb8a1e32b528fef8bfd677d2b3fc5ab37278b157fa...</td>
      <td>JC325941</td>
      <td>NaN</td>
      <td>06/27/2019 04:00:00 PM</td>
      <td>45</td>
      <td>NO CONTROLS</td>
      <td>NO CONTROLS</td>
      <td>CLEAR</td>
      <td>DAYLIGHT</td>
      <td>SIDESWIPE SAME DIRECTION</td>
      <td>...</td>
      <td>NaN</td>
      <td>UNKNOWN</td>
      <td>UNKNOWN</td>
      <td>NORMAL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TEST NOT OFFERED</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>009e9e67203442370272e1a13d6ee51a4155dac65e583d...</td>
      <td>JA329216</td>
      <td>NaN</td>
      <td>06/30/2017 04:00:00 PM</td>
      <td>35</td>
      <td>STOP SIGN/FLASHER</td>
      <td>FUNCTIONING PROPERLY</td>
      <td>CLEAR</td>
      <td>DAYLIGHT</td>
      <td>TURNING</td>
      <td>...</td>
      <td>NaN</td>
      <td>FAILED TO YIELD</td>
      <td>UNKNOWN</td>
      <td>UNKNOWN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TEST NOT OFFERED</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>009e9e67203442370272e1a13d6ee51a4155dac65e583d...</td>
      <td>JA329216</td>
      <td>NaN</td>
      <td>06/30/2017 04:00:00 PM</td>
      <td>35</td>
      <td>STOP SIGN/FLASHER</td>
      <td>FUNCTIONING PROPERLY</td>
      <td>CLEAR</td>
      <td>DAYLIGHT</td>
      <td>TURNING</td>
      <td>...</td>
      <td>NaN</td>
      <td>NONE</td>
      <td>NOT OBSCURED</td>
      <td>NORMAL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TEST NOT OFFERED</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 78 columns</p>
</div>




```python
# A quick preview of the data
display(crash_df.info())
crash_df.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 983415 entries, 0 to 983414
    Data columns (total 78 columns):
     #   Column                         Non-Null Count   Dtype  
    ---  ------                         --------------   -----  
     0   CRASH_RECORD_ID                983415 non-null  object 
     1   RD_NO_x                        975129 non-null  object 
     2   CRASH_DATE_EST_I               55419 non-null   object 
     3   CRASH_DATE_x                   983415 non-null  object 
     4   POSTED_SPEED_LIMIT             983415 non-null  int64  
     5   TRAFFIC_CONTROL_DEVICE         983415 non-null  object 
     6   DEVICE_CONDITION               983415 non-null  object 
     7   WEATHER_CONDITION              983415 non-null  object 
     8   LIGHTING_CONDITION             983415 non-null  object 
     9   FIRST_CRASH_TYPE               983415 non-null  object 
     10  TRAFFICWAY_TYPE                983415 non-null  object 
     11  LANE_CNT                       446675 non-null  float64
     12  ALIGNMENT                      983415 non-null  object 
     13  ROADWAY_SURFACE_COND           983415 non-null  object 
     14  ROAD_DEFECT                    983415 non-null  object 
     15  REPORT_TYPE                    954194 non-null  object 
     16  CRASH_TYPE                     983415 non-null  object 
     17  INTERSECTION_RELATED_I         264399 non-null  object 
     18  NOT_RIGHT_OF_WAY_I             35715 non-null   object 
     19  HIT_AND_RUN_I                  242928 non-null  object 
     20  DAMAGE                         983415 non-null  object 
     21  DATE_POLICE_NOTIFIED           983415 non-null  object 
     22  PRIM_CONTRIBUTORY_CAUSE        983415 non-null  object 
     23  SEC_CONTRIBUTORY_CAUSE         983415 non-null  object 
     24  STREET_NO                      983415 non-null  int64  
     25  STREET_DIRECTION               983410 non-null  object 
     26  STREET_NAME                    983413 non-null  object 
     27  BEAT_OF_OCCURRENCE             983403 non-null  float64
     28  PHOTOS_TAKEN_I                 12866 non-null   object 
     29  STATEMENTS_TAKEN_I             23004 non-null   object 
     30  DOORING_I                      3183 non-null    object 
     31  WORK_ZONE_I                    6462 non-null    object 
     32  WORK_ZONE_TYPE                 5164 non-null    object 
     33  WORKERS_PRESENT_I              1493 non-null    object 
     34  NUM_UNITS                      983415 non-null  int64  
     35  MOST_SEVERE_INJURY             982532 non-null  object 
     36  INJURIES_TOTAL                 982542 non-null  float64
     37  INJURIES_FATAL                 982542 non-null  float64
     38  INJURIES_INCAPACITATING        982542 non-null  float64
     39  INJURIES_NON_INCAPACITATING    982542 non-null  float64
     40  INJURIES_REPORTED_NOT_EVIDENT  982542 non-null  float64
     41  INJURIES_NO_INDICATION         982542 non-null  float64
     42  INJURIES_UNKNOWN               982542 non-null  float64
     43  CRASH_HOUR                     983415 non-null  int64  
     44  CRASH_DAY_OF_WEEK              983415 non-null  int64  
     45  CRASH_MONTH                    983415 non-null  int64  
     46  LATITUDE                       977913 non-null  float64
     47  LONGITUDE                      977913 non-null  float64
     48  LOCATION                       977913 non-null  object 
     49  PERSON_ID                      982540 non-null  object 
     50  PERSON_TYPE                    982540 non-null  object 
     51  RD_NO_y                        982539 non-null  object 
     52  VEHICLE_ID                     962876 non-null  float64
     53  CRASH_DATE_y                   982540 non-null  object 
     54  SEAT_NO                        202307 non-null  float64
     55  CITY                           728774 non-null  object 
     56  STATE                          736767 non-null  object 
     57  ZIPCODE                        666120 non-null  object 
     58  SEX                            967956 non-null  object 
     59  AGE                            704186 non-null  float64
     60  DRIVERS_LICENSE_STATE          583538 non-null  object 
     61  DRIVERS_LICENSE_CLASS          509291 non-null  object 
     62  SAFETY_EQUIPMENT               979605 non-null  object 
     63  AIRBAG_DEPLOYED                963642 non-null  object 
     64  EJECTION                       970329 non-null  object 
     65  INJURY_CLASSIFICATION          982000 non-null  object 
     66  HOSPITAL                       180236 non-null  object 
     67  EMS_AGENCY                     115037 non-null  object 
     68  EMS_RUN_NO                     18739 non-null   object 
     69  DRIVER_ACTION                  778324 non-null  object 
     70  DRIVER_VISION                  778081 non-null  object 
     71  PHYSICAL_CONDITION             778903 non-null  object 
     72  PEDPEDAL_ACTION                18564 non-null   object 
     73  PEDPEDAL_VISIBILITY            18521 non-null   object 
     74  PEDPEDAL_LOCATION              18565 non-null   object 
     75  BAC_RESULT                     779427 non-null  object 
     76  BAC_RESULT VALUE               1254 non-null    float64
     77  CELL_PHONE_USE                 1156 non-null    object 
    dtypes: float64(15), int64(6), object(57)
    memory usage: 592.7+ MB



### Column Names and Descriptions

**POSTED_SPEED_LIMIT**	
Posted speed limit, as determined by reporting officer

**WEATHER_CONDITION**	
Weather condition at time of crash, as determined by reporting officer

**LIGHTING_CONDITION**	
Light condition at time of crash, as determined by reporting officer

**FIRST_CRASH_TYPE**	
Type of first collision in crash

**TRAFFICWAY_TYPE**	
Trafficway type, as determined by reporting officer

**LANE_CNT**	
Total number of through lanes in either direction, excluding turn lanes, as determined by reporting officer (0 = intersection)

**ALIGNMENT**	
Street alignment at crash location, as determined by reporting officer

**ROADWAY_SURFACE_COND**	
Road surface condition, as determined by reporting officer

**ROAD_DEFECT**	
Road defects, as determined by reporting officer

**CRASH_TYPE**	
A general severity classification for the crash. Can be either Injury and/or Tow Due to Crash or No Injury / Drive Away

**INTERSECTION_RELATED_I**	
A field observation by the police officer whether an intersection played a role in the crash. Does not represent whether or not the crash occurred within the intersection.

**NOT_RIGHT_OF_WAY_I**	
Whether the crash begun or first contact was made outside of the public right-of-way.

**HIT_AND_RUN_I**
Crash did/did not involve a driver who caused the crash and fled the scene without exchanging information and/or rendering aid

**DAMAGE**	
A field observation of estimated damage.

**PRIM_CONTRIBUTORY_CAUSE**	
The factor which was most significant in causing the crash, as determined by officer judgment

**SEC_CONTRIBUTORY_CAUSE**	
The factor which was second most significant in causing the crash, as determined by officer judgment

**BEAT_OF_OCCURRENCE**	
Chicago Police Department Beat ID. Boundaries available at https://data.cityofchicago.org/d/aerh-rz74

**STATEMENTS_TAKEN_I**	
Whether statements were taken from unit(s) involved in crash

**WORK_ZONE_TYPE**	
The type of work zone, if any

**NUM_UNITS**	
Number of units involved in the crash. A unit can be a motor vehicle, a pedestrian, a bicyclist, or another non-passenger roadway user. Each unit represents a mode of traffic with an independent trajectory.

**CRASH_HOUR**	
The hour of the day component of CRASH_DATE.

**CRASH_DAY_OF_WEEK**	
The day of the week component of CRASH_DATE. Sunday=1

**CRASH_MONTH**	
The month component of CRASH_DATE.

**SEX**	
Gender of person involved in crash, as determined by reporting officer

**AGE**	
Age of person involved in crash

**AIRBAG_DEPLOYED**	
Whether vehicle occupant airbag deployed as result of crash

**EJECTION**	
Whether vehicle occupant was ejected or extricated from the vehicle as a result of crash

**INJURY_CLASSIFICATION**	
Severity of injury person sustained in the crash

**DRIVER_ACTION**	
Driver action that contributed to the crash, as determined by reporting officer

**DRIVER_VISION**	
What, if any, objects obscured the driver’s vision at time of crash

**PHYSICAL_CONDITION**	
Driver’s apparent physical condition at time of crash, as observed by the reporting officer

**PEDPEDAL_VISIBILITY**	
Visibility of pedestrian of cyclist safety equipment in use at time of crash

**BAC_RESULT**	
Status of blood alcohol concentration testing for driver or other person involved in crash

**BAC_RESULT VALUE**	
Driver’s blood alcohol concentration test result (fatal crashes may include pedestrian or cyclist results)

**CELL_PHONE_USE**	
Whether person was/was not using cellphone at the time of the crash, as determined by the reporting officer


## Scrubbing and exploring the data

The process of scrubbing the dataset included: 

1. Removing all duplicate crashes based on the unique crash_ID

2. Dropping columns that would not be beneficial to the model or had similar data to other columns used.

3. Dealing with null values
   - Object null values were either dropped, filled with their verison of 'NO', or filled with "UNKNOWN" based on the unique values in each column.
   - Numerical null values were either dropped, filled with 0, filled with the mean, or filled with the mode based on the unique values and make up of each column.
   
4. Once all the scrubbing was completed, the final shape of the dataframe was (442956, 34)


The process of exploring the dataset included:

1. Creating functions to streamline plotting data.

```python
# Function to graph bar chart
def bar_plot(col, target, df):
    """Creates bar plot against target variable and saves plot as png
    to figures folder"""
    ax = sns.barplot(y=df[target], x= df[col], orient='h', data=df)
    ax.set_title(f"{col} vs {target}")

    # Save image as png
    plt.savefig(f'figures/bar_{col}_v_{target}.png', transparent=True, bbox_inches='tight');
```


```python
# Function to graph a count plot
def count_plot(df,col):
    """Creates boxplot of a column and saves plot as png to figures folder"""
    ax = sns.countplot(x=df[col], data=df)
    ax.set_title(f"{col}")
    plt.xticks(rotation=270)
    
    # Save image as png
    plt.savefig(f'figures/count_{col}.png', transparent=True, bbox_inches='tight');
```


```python
def scatter_plot(col,target,df):
    """Creates a scatterplot of a column against target variable with a hue of
    crash type and saves plot to figures folder."""
    
    ax = sns.scatterplot(data=df, x=df[col], y=df[target])
    ax.set_title(f"{col} v {target}")
     # Save image as png
    plt.savefig(f'figures/scatter_{col}_v_{target}.png', transparent=True, bbox_inches='tight');
```

2. Looking at count plots for the categorical columns. This is where it was decided to focus on Driver_Action. 


![countplot driver action](Figures/count_DRIVER_ACTION.png)



3. Driver_action was One-Hot Encoded and the other object columns were LabelEncoded


```python
feats = ['DRIVER_ACTION']
```


```python
encode_df = pd.get_dummies(crash_df, drop_first=True, columns=feats)
encode_df.info()
```


```python
# LabelEncode Feature Columns onto a new df to examine against our target
from sklearn.preprocessing import LabelEncoder

# creating instance of labelencoder
labelencoder = LabelEncoder()

# Using for loop to loop through object_col and Assign numerical values 
# and storing in another column
for col in object_col:
    encode_df[col] = labelencoder.fit_transform(encode_df[col])

encode_df.head()
```


4. Histograms of numerical columns were made.


```python
# Histograms 
encode_df.hist(figsize = (30,20))
plt.savefig('Figures/crash_df_hist.png', dpi=300, bbox_inches='tight');
```


![histogram](Figures/crash_df_hist.png)



5. Barplots and scatterplots were made to get a better idea of the breakdown of the data and helped narrow down the target to Failed to yield. 


```python
# Bar graph of the Crash Type

plt.figure(figsize=(15,15))

y= fail_yield.FIRST_CRASH_TYPE.value_counts().values
x=fail_yield.FIRST_CRASH_TYPE.value_counts().index

sns.barplot(y, x)
plt.title('Fail to Yield First Crash Type', size=30)
plt.ylabel("Crash Type", size=25, rotation=90)
plt.xlabel("Quantity", size=25)
plt.xticks(size=15)
plt.yticks(size=15)

plt.savefig('Figures/Fail_yield_first_crash.png', dpi=300, bbox_inches='tight');
```


![barplot failed to yield crash type](Figures/Fail_yield_first_crash.png)



```python
# Bar graph of the Crash Type

plt.figure(figsize=(15,15))

x= fail_yield.CRASH_HOUR.value_counts().values
y=fail_yield.CRASH_HOUR.value_counts().index

sns.barplot(y, x)
plt.title('Fail to Yield CRASH HOUR', size=30)
plt.ylabel("Quantity", size=25, rotation=90)
plt.xlabel("Hour", size=25)
plt.xticks(size=15)
plt.yticks(size=15)

plt.savefig('Figures/Fail_yield_crash_hour.png', dpi=300, bbox_inches='tight');
```


![Failed to yield crash hour](Figures/Fail_yield_crash_hour.png)



```python
# Bar graph of the Crash Type

plt.figure(figsize=(15,15))

x= fail_yield.DRIVER_VISION.value_counts().values
y=fail_yield.DRIVER_VISION.value_counts().index

sns.barplot(y, x)
plt.title('Fail to Yield DRIVER VISION', size=30)
plt.ylabel("Quantity", size=25, rotation=90)
plt.xlabel("", size=25)
plt.xticks(size=15, rotation=90)
plt.yticks(size=15)

plt.savefig('Figures/Fail_yield_driver_vision.png', dpi=300, bbox_inches='tight');
```


![failed to yield drivers vision](Figures/Fail_yield_driver_vision.png)


## Models for Target: FAILED TO YIELD



Functions to help ease the process of creating multiple models. 


```python
# Function to print scores
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
def print_metrics(labels, preds, time):
    """input (labels, preds) and prints Precision score, Recall score,
    Accuracy score, F1 score, and Confusion matrix"""
    print("Precision Score: {}".format(precision_score(labels, preds)))
    print("Recall Score: {}".format(recall_score(labels, preds)))
    print("Accuracy Score: {}".format(accuracy_score(labels, preds)))
    print("F1 Score: {}".format(f1_score(labels, preds)))
    print("Confision Matrix: \n{}".format(confusion_matrix(labels, preds)))
    print("\n Run Time: {}".format(time))
```


```python
def print_tree(dt, feature_name, save_file):
    """function to print and save the Decision Tree printout"""
    dot_data = tree.export_graphviz(dt, 
                  feature_names=feature_name,    
                  filled=True, rounded=True,  
                  special_characters=True,
                  out_file=None)
            
    graph = graphviz.Source(dot_data)

    graph.format = "png"
    graph.render(f'figures/{save_file}')
    return graph.view()
```


```python
def plot_feature_importances(model):
    """Plots feature importance"""
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), X_train.columns.values) 
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
```


THe data was then Train-Test_Split and scaled using StandardScaler





### Baseline model: Dummy Classifier

Dummy Classifier was used as the baseline model due to its speed and simplicity. It makes predictions based on simple rules. In this case, it made prediction based on class by using the strategy 'stratified.'


```python
# generates random predictions by respecting the training set class distribution.
from sklearn.dummy import DummyClassifier
```


```python
start = time.time()

# Instantiate 
clf = DummyClassifier(strategy='stratified', random_state=0)

# Fit the Classifier
clf.fit(X_train, y_train)
DummyClassifier(random_state=0, strategy='stratified')
clf.score(X_test, y_test)

# Predict on the test set
y_preds = clf.predict(X_test)

end = time.time()
dum_n = end - start
print(f'time: {dum_n}')
```

The scores with the Dummy Classifier are low, which was to be expected. There is a very high false positive rate. 

```python
print_metrics(y_preds, y_test, dum_n)
```

    Precision Score: 0.14946210564486884
    Recall Score: 0.15072834765648807
    Accuracy Score: 0.7469826184367434
    F1 Score: 0.15009255606469818
    Confision Matrix: 
    [[80212 14073]
     [13934  2473]]
    
     Run Time: 0.05151510238647461


### Final model: Decision Tree

Decision trees break data into smaller subsets based on feature importance. Unlike distance based models with a dataset this large, it will not take hours to run.


![dt flowchart](Features/dt_printout.png)


```python
plot_feature_importances(dt_model)
```

![feature importance plot](Figures/feature_importance.png)





```python
start = time.time()

# Train a Decesion Tree
dt_model = DecisionTreeClassifier(random_state=10, max_depth=5)  
dt_model.fit(X_train, y_train) 

# Make predictions for test data
y_pred = dt_model.predict(X_test)

end = time.time()
dt_n = end - start
dt_n
```

The results of this model were much better than the baseline model, but still not great. 

```python
print_metrics(y_pred, y_test, dt_n)
```

    Precision Score: 0.5809259035416415
    Recall Score: 0.7744742567077593
    Accuracy Score: 0.9120713330683338
    F1 Score: 0.6638809268915979
    Confision Matrix: 
    [[91347  6934]
     [ 2799  9612]]
    
     Run Time: 1.4316511154174805
     
     
### Recommendations

In driver education classes:
- Focus more on the importance of yielding especially when turning, changing lanes, and with pedestrians present

- Rush hour when everyone is in a rush to get to work/get home is the time to be extra cautious 

For filling in crash reports:
 - Since most of the data is based on the determination of the reporting officer, having universal language would help repetitive data and multiple inputs meaning the same thing. ie the Hospital column having over 25 ways that "Refused Medical Attention" was inputted. 
 
 
 
#### Recommended Future Work 

- Examine how other action and causes affect the models
- Add the third linked dataframe covering the cars involved to go more in depth.

- Streamline the data:
 - Research the differences in crashes, causes, etc. Example: not yielding to parked vehicle? 
 - Combine like columns, rows based on the research
 - Binning and cluster data 
- Run other models that take longer to run




