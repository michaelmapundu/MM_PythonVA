

print('importing required packages...')
import pandas as pd, numpy as np, os, re, time, matplotlib.pyplot as plt, seaborn as sns, warnings, psycopg2
from sqlalchemy import create_engine
from collections import Counter
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from yellowbrick.classifier import ROCAUC
from yellowbrick.datasets import load_game
from matplotlib import pyplot
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.combine import SMOTETomek
import statsmodels.api as sm
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB

#from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

#from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, label_binarize
'''Metrics/Evaluation'''
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

'''Display'''
from IPython.core.display import display, HTML
print('required packages loaded and view set successfully...')

#seting a few parameters
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)
#pd.set_option('display.max_colwidth', -13)
pd.options.display.float_format = '{:,.2f}'.format
display(HTML("<style>.container { width:95% !important; }</style>"))
sns.set_style('darkgrid')
warnings.filterwarnings('ignore')

start = time.time()
startptd = time.strftime('%X %x %Z')
print('\n','The program start time and Date','\n',startptd)

#Setting the working directory
os.chdir('C:\\Users\\a0056407\\Desktop\\Michael_Mapundu__Docs\\PhDPythonScript\\FinalMLVasOnly')
df = pd.read_csv('C:\\Users\\a0056407\\Desktop\\Michael_Mapundu__Docs\\PhDPythonScript\\FinalMLVasOnly\\MichaelData.csv', encoding='latin1')
print('data read successfully and connection closed...')
#Converting the columns headers to lower case
df.columns = df.columns.str.lower()
conn = psycopg2.connect(user="postgres", password="postgres",
                                  host="127.0.0.1", port="5432", database="mike")
print('deleting table from database if exist')
with conn:
    cursor = conn.cursor()
    cursor.execute('drop table if exists encryted_ncr')
    conn.commit()

print('exporting the dataframe to postgres table...')
engine = create_engine('postgresql://postgres:postgres@localhost:5432/mike')
df.to_sql('cleaned_reports', engine, chunksize=10000, index=False)
df.to_sql('verbal_autopsy', engine, chunksize=10000, index=False)
print('data loaded to postgres successfully...')
cursor = conn.cursor()

sql_cmd1 = "select id, disease_description from cleaned_reports"
sql_cmd2 = "select id, icdmain_consensus from verbal_autopsy"
cursor.execute(sql_cmd1)
cursor.execute(sql_cmd2)
df1 = pd.read_sql(sql_cmd1, conn)
df2 = pd.read_sql(sql_cmd2, conn)
#closing the connection
cursor.close()
conn.close()
print('data read successfully and connection closed...')

print('merge the two datasets...')
dfA = pd.merge(df1, df2, on='id')

print(len(dfA))
print(dfA.head())

print('drop nulls...')
dfA_subA = dfA.dropna(subset=['icdmain_consensus'])
dfA_subA["icdmain_consensus"] = dfA_subA["icdmain_consensus"].str.upper()
dfA_subA['length'] = dfA_subA['icdmain_consensus'].str.len()
dfA_subA = dfA_subA[dfA_subA.length > 2]
print(len(dfA_subA))
print(dfA_subA.head())
del dfA_subA['length']

print('record the cause of death categories...')
dfA_subA['broad_cause_cat'] = ''
dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'B20') | (dfA_subA['icdmain_consensus'] == 'B24')|
             (dfA_subA['icdmain_consensus'] == 'A16') | (dfA_subA['icdmain_consensus'] == 'A15'),
             'broad_cause_cat'] = 'HIV/AIDS & TB'  

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'A40') | (dfA_subA['icdmain_consensus'] == 'A41')|
             (dfA_subA['icdmain_consensus'] == 'J00') | (dfA_subA['icdmain_consensus'] == 'J22')|
             (dfA_subA['icdmain_consensus'] == 'A00') | (dfA_subA['icdmain_consensus'] == 'A09')|
             (dfA_subA['icdmain_consensus'] == 'B50') | (dfA_subA['icdmain_consensus'] == 'B54')|
             (dfA_subA['icdmain_consensus'] == 'B05') | (dfA_subA['icdmain_consensus'] == 'A39')|
             (dfA_subA['icdmain_consensus'] == 'G00') | (dfA_subA['icdmain_consensus'] == 'G05')|
             (dfA_subA['icdmain_consensus'] == 'A33') | (dfA_subA['icdmain_consensus'] == 'A35')|
             (dfA_subA['icdmain_consensus'] == 'A37') | (dfA_subA['icdmain_consensus'] == 'A92')|
             (dfA_subA['icdmain_consensus'] == 'A99') | (dfA_subA['icdmain_consensus'] == 'A90')|
             (dfA_subA['icdmain_consensus'] == 'A91') | (dfA_subA['icdmain_consensus'] == 'A17')|
             (dfA_subA['icdmain_consensus'] == 'A19') | (dfA_subA['icdmain_consensus'] == 'A20')|
             (dfA_subA['icdmain_consensus'] == 'A38') | (dfA_subA['icdmain_consensus'] == 'A42')|
             (dfA_subA['icdmain_consensus'] == 'A89') | (dfA_subA['icdmain_consensus'] == 'B00')|
             (dfA_subA['icdmain_consensus'] == 'B19') | (dfA_subA['icdmain_consensus'] == 'B25')|
             (dfA_subA['icdmain_consensus'] == 'B49') | (dfA_subA['icdmain_consensus'] == 'B55')|
             (dfA_subA['icdmain_consensus'] == 'B99'),'broad_cause_cat'] = 'Other infectious'  

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'O00') | (dfA_subA['icdmain_consensus'] == 'O03')|
             (dfA_subA['icdmain_consensus'] == 'O08') | (dfA_subA['icdmain_consensus'] == 'O10')|
             (dfA_subA['icdmain_consensus'] == 'O16') | (dfA_subA['icdmain_consensus'] == 'O46')|
             (dfA_subA['icdmain_consensus'] == 'O67') | (dfA_subA['icdmain_consensus'] == 'O72')|
             (dfA_subA['icdmain_consensus'] == 'O63') | (dfA_subA['icdmain_consensus'] == 'O66')|
             (dfA_subA['icdmain_consensus'] == 'O75.3') | (dfA_subA['icdmain_consensus'] == 'O85')|
             (dfA_subA['icdmain_consensus'] == 'O99.0') | (dfA_subA['icdmain_consensus'] == 'O71')|
             (dfA_subA['icdmain_consensus'] == 'P95') | (dfA_subA['icdmain_consensus'] == 'P36')|
             (dfA_subA['icdmain_consensus'] == 'A33') | (dfA_subA['icdmain_consensus'] == 'Q00')|
             (dfA_subA['icdmain_consensus'] == 'Q99') | (dfA_subA['icdmain_consensus'] == 'P00')|
             (dfA_subA['icdmain_consensus'] == 'P04') | (dfA_subA['icdmain_consensus'] == 'P08')|
             (dfA_subA['icdmain_consensus'] == 'P15') | (dfA_subA['icdmain_consensus'] == 'P26')|
             (dfA_subA['icdmain_consensus'] == 'P35') | (dfA_subA['icdmain_consensus'] == 'P37')|
             (dfA_subA['icdmain_consensus'] == 'P94') | (dfA_subA['icdmain_consensus'] == 'P96')|
             (dfA_subA['icdmain_consensus'] == 'P05') | (dfA_subA['icdmain_consensus'] == 'P07')|
             (dfA_subA['icdmain_consensus'] == 'P20') | (dfA_subA['icdmain_consensus'] == 'P22')|
             (dfA_subA['icdmain_consensus'] == 'P23') | (dfA_subA['icdmain_consensus'] == 'P25')|
             (dfA_subA['icdmain_consensus'] == 'O01') | (dfA_subA['icdmain_consensus'] == 'O02')|
             (dfA_subA['icdmain_consensus'] == 'O20') | (dfA_subA['icdmain_consensus'] == 'O45')|
             (dfA_subA['icdmain_consensus'] == 'O47') | (dfA_subA['icdmain_consensus'] == 'O62')|
             (dfA_subA['icdmain_consensus'] == 'O68') | (dfA_subA['icdmain_consensus'] == 'O70')|
             (dfA_subA['icdmain_consensus'] == 'O73') | (dfA_subA['icdmain_consensus'] == 'O84')|
             (dfA_subA['icdmain_consensus'] == 'O86') | (dfA_subA['icdmain_consensus'] == 'O99'),
             'broad_cause_cat'] = 'Martenal & neonatal'  

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'V01') | (dfA_subA['icdmain_consensus'] == 'V89')|
             (dfA_subA['icdmain_consensus'] == 'V99') | (dfA_subA['icdmain_consensus'] == 'V90')|
             (dfA_subA['icdmain_consensus'] == 'W00') | (dfA_subA['icdmain_consensus'] == 'W19')|
             (dfA_subA['icdmain_consensus'] == 'W65') | (dfA_subA['icdmain_consensus'] == 'W74')|
             (dfA_subA['icdmain_consensus'] == 'X00') | (dfA_subA['icdmain_consensus'] == 'X19')|
             (dfA_subA['icdmain_consensus'] == 'X20') | (dfA_subA['icdmain_consensus'] == 'X29')|
             (dfA_subA['icdmain_consensus'] == 'X40') | (dfA_subA['icdmain_consensus'] == 'X49')|
             (dfA_subA['icdmain_consensus'] == 'X60') | (dfA_subA['icdmain_consensus'] == 'X84')|
             (dfA_subA['icdmain_consensus'] == 'X85') | (dfA_subA['icdmain_consensus'] == 'Y09')|
             (dfA_subA['icdmain_consensus'] == 'X30') | (dfA_subA['icdmain_consensus'] == 'X39')|
             (dfA_subA['icdmain_consensus'] == 'S00') | (dfA_subA['icdmain_consensus'] == 'T99')|
             (dfA_subA['icdmain_consensus'] == 'W20') | (dfA_subA['icdmain_consensus'] == 'W64')|
             (dfA_subA['icdmain_consensus'] == 'W75') | (dfA_subA['icdmain_consensus'] == 'W99')|
             (dfA_subA['icdmain_consensus'] == 'X50') | (dfA_subA['icdmain_consensus'] == 'X59')|
             (dfA_subA['icdmain_consensus'] == 'Y10') | (dfA_subA['icdmain_consensus'] == 'Y98'),
             'broad_cause_cat'] = 'External causes'  

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'J40') | (dfA_subA['icdmain_consensus'] == 'J44')|
             (dfA_subA['icdmain_consensus'] == 'J45') | (dfA_subA['icdmain_consensus'] == 'J46'),
             'broad_cause_cat'] = 'Respiratory'  

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'R10') | (dfA_subA['icdmain_consensus'] == 'K70')|
             (dfA_subA['icdmain_consensus'] == 'K76') | (dfA_subA['icdmain_consensus'] == 'N17')|
             (dfA_subA['icdmain_consensus'] == 'N19'), 'broad_cause_cat'] = 'Abdominal'  

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'G40') | (dfA_subA['icdmain_consensus'] == 'G41'),
             'broad_cause_cat'] = 'Neurological' 

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'R95') | (dfA_subA['icdmain_consensus'] == 'R99'),
             'broad_cause_cat'] = 'Indeterminate'

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'D50') | (dfA_subA['icdmain_consensus'] == 'D64')|
             (dfA_subA['icdmain_consensus'] == 'E40') | (dfA_subA['icdmain_consensus'] == 'E46')|
             (dfA_subA['icdmain_consensus'] == 'E10') | (dfA_subA['icdmain_consensus'] == 'E14')
             ,'broad_cause_cat'] = 'Metabolic'  

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'I20') | (dfA_subA['icdmain_consensus'] == 'I25')|
             (dfA_subA['icdmain_consensus'] == 'I60') | (dfA_subA['icdmain_consensus'] == 'I69')|
             (dfA_subA['icdmain_consensus'] == 'D57') | (dfA_subA['icdmain_consensus'] == 'I00')|
             (dfA_subA['icdmain_consensus'] == 'I09') | (dfA_subA['icdmain_consensus'] == 'I10')|
             (dfA_subA['icdmain_consensus'] == 'I15') | (dfA_subA['icdmain_consensus'] == 'I26')|
             (dfA_subA['icdmain_consensus'] == 'I52') | (dfA_subA['icdmain_consensus'] == 'I70')|
             (dfA_subA['icdmain_consensus'] == 'I99'),'broad_cause_cat'] = 'Cardiovascular' 

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'C00') | (dfA_subA['icdmain_consensus'] == 'C06')|
             (dfA_subA['icdmain_consensus'] == 'C15') | (dfA_subA['icdmain_consensus'] == 'C26')|
             (dfA_subA['icdmain_consensus'] == 'C30') | (dfA_subA['icdmain_consensus'] == 'C39')|
             (dfA_subA['icdmain_consensus'] == 'C50') | (dfA_subA['icdmain_consensus'] == 'C51')|
             (dfA_subA['icdmain_consensus'] == 'C58') | (dfA_subA['icdmain_consensus'] == 'C60')|
             (dfA_subA['icdmain_consensus'] == 'C63') | (dfA_subA['icdmain_consensus'] == 'C07')|
             (dfA_subA['icdmain_consensus'] == 'C14') | (dfA_subA['icdmain_consensus'] == 'C40')|
             (dfA_subA['icdmain_consensus'] == 'C49') | (dfA_subA['icdmain_consensus'] == 'C48'),
             'broad_cause_cat'] = 'Neoplasms' 

dfA_subA.loc[(dfA_subA['icdmain_consensus'] == 'M99') | (dfA_subA['icdmain_consensus'] == 'N00')|
             (dfA_subA['icdmain_consensus'] == 'N16') | (dfA_subA['icdmain_consensus'] == 'N20')|
             (dfA_subA['icdmain_consensus'] == 'N99') | (dfA_subA['icdmain_consensus'] == 'R00')|
             (dfA_subA['icdmain_consensus'] == 'R09') | (dfA_subA['icdmain_consensus'] == 'R11')|
             (dfA_subA['icdmain_consensus'] == 'R94') | (dfA_subA['icdmain_consensus'] == 'D55')|
             (dfA_subA['icdmain_consensus'] == 'D89') | (dfA_subA['icdmain_consensus'] == 'E00')|
             (dfA_subA['icdmain_consensus'] == 'E07') | (dfA_subA['icdmain_consensus'] == 'E15')|
             (dfA_subA['icdmain_consensus'] == 'E35') | (dfA_subA['icdmain_consensus'] == 'E50')|
             (dfA_subA['icdmain_consensus'] == 'E90') | (dfA_subA['icdmain_consensus'] == 'F00')|
             (dfA_subA['icdmain_consensus'] == 'F99') | (dfA_subA['icdmain_consensus'] == 'G06')|
             (dfA_subA['icdmain_consensus'] == 'G09') | (dfA_subA['icdmain_consensus'] == 'G10')|
             (dfA_subA['icdmain_consensus'] == 'G37') | (dfA_subA['icdmain_consensus'] == 'G50')|
             (dfA_subA['icdmain_consensus'] == 'G99') | (dfA_subA['icdmain_consensus'] == 'H00')|
             (dfA_subA['icdmain_consensus'] == 'H95') | (dfA_subA['icdmain_consensus'] == 'J30')|
             (dfA_subA['icdmain_consensus'] == 'J99') | (dfA_subA['icdmain_consensus'] == 'K00')|
             (dfA_subA['icdmain_consensus'] == 'K31') | (dfA_subA['icdmain_consensus'] == 'K35')|
             (dfA_subA['icdmain_consensus'] == 'K38') | (dfA_subA['icdmain_consensus'] == 'K40')|
             (dfA_subA['icdmain_consensus'] == 'K93') | (dfA_subA['icdmain_consensus'] == 'L00')|
             (dfA_subA['icdmain_consensus'] == 'L99') | (dfA_subA['icdmain_consensus'] == 'M00')|
             (dfA_subA['icdmain_consensus'] == 'J39') | (dfA_subA['icdmain_consensus'] == 'J47'),
             'broad_cause_cat'] = 'Other NCD'  


#Checking for class distribution
print(dfA_subA['broad_cause_cat'].value_counts())

#IMBALANCED
data = dfA_subA.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# label encode the target variable
y = LabelEncoder().fit_transform(y)
# summarize distribution
counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()

#keep the two categories
#dfA_subB = dfA_subA[(dfA_subA['broad_cause_cat'] == 'HIV/AIDS & TB') | 
#                    (dfA_subA['broad_cause_cat'] == 'Other infectious')]
dfA_subB = dfA_subA[(dfA_subA['broad_cause_cat'] != '')]

print('encoding the labels in the dataset, please wait...')
#Turning labels into numbers
LE = LabelEncoder()
dfA_subB['label_num'] = LE.fit_transform(dfA_subB['broad_cause_cat'])
print(dfA_subB.head())

print('calling tfidf vectorizor, please wait...')
#tfidf vectorizer with 1 and 2 grams
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df = 2, max_df = .95)

print('generating features using tfidf vectorizor, please wait...')
#splitting the data into features and targets
X = tfidf_vectorizer.fit_transform(dfA_subB['disease_description'].values.astype(str)) #features
y = dfA_subB['label_num'].values #target
print (X.shape)
print(y.shape)

df = pd.DataFrame(X)
df['target'] = y
plt.figure(figsize=(12, 8))
df.target.value_counts().plot(kind='bar', title='Count (target)')
plt.ylabel('Counts', fontsize = 13)
plt.xlabel('classes', fontsize = 13)
plt.savefig('classes_before_balancing.png', dpi = 900)
plt.savefig('classes_before_balancing.pdf', dpi = 900)
plt.show()

print('balancing the data')
smote = SMOTE('minority')
# transform the dataset
strategy = {0:3388, 1:3388, 2:3388, 3:3388, 4:3388, 5:3388, 6:3388, 7:3388, 8:3388, 9:3388, 10:3388, 11:3388}
oversample = SMOTE(sampling_strategy=strategy)
X, y = oversample.fit_resample(X, y)

# summarize distribution
counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()
#smote = SMOTE('minority')
#X, y = smote.fit_sample(X, y)

df = pd.DataFrame(X)
df['target'] = y
plt.figure(figsize=(12, 8))
df.target.value_counts().plot(kind='bar', title='Count (target)')
plt.ylabel('Counts', fontsize = 13)
plt.xlabel('classes', fontsize = 13)
plt.savefig('classes_after_balancing.png', dpi = 900)
plt.savefig('classes_after_balancing.pdf', dpi = 900)
plt.show()


print('reducing the features to best 100, please wait...')
#Dimenionality reduction. Only using the 100 best features
lsa = TruncatedSVD(n_components=100, n_iter=10, random_state=3)
X = lsa.fit_transform(X)
print(X.shape)


#Preliminary model evaluation using default parameters
print('generating algorithms performance before optimization, please wait...')
#Creating a dict of the models
model_dict = {'Random Forest': RandomForestClassifier(random_state=3),
              'NBC': MultinomialNB(),
              'Logistic Regression':LogisticRegression(),
              'K Nearest Neighbor': KNeighborsClassifier(),
              'ANN': MLPClassifier(),
               'XGBoost': XGBClassifier(),
              'DT':DecisionTreeClassifier(),
             'Support Vector Machine': SVC()}

#Train test split with stratified sampling for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = .3, 
                                                    shuffle = True, 
                                                    stratify = y, 
                                                    random_state = 3)

from sklearn.preprocessing import MinMaxScaler #fixed import

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Function to get the scores for each model in a df
def model_score_df(model_dict):   
    model_name, ac_score_list, p_score_list, r_score_list, f1_score_list = [], [], [], [], []
    for k,v in model_dict.items():   
        model_name.append(k)
        v.fit(X_train, y_train)
        y_pred = v.predict(X_test)
        ac_score_list.append(accuracy_score(y_test, y_pred))
        p_score_list.append(precision_score(y_test, y_pred, average='macro'))
        r_score_list.append(recall_score(y_test, y_pred, average='macro'))
        f1_score_list.append(f1_score(y_test, y_pred, average='macro'))
        model_comparison_df = pd.DataFrame([model_name, ac_score_list, p_score_list, r_score_list, f1_score_list]).T
        model_comparison_df.columns = ['model_name', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']
        model_comparison_df = model_comparison_df.sort_values(by='f1_score', ascending=False)
    return model_comparison_df

#print(model_score_df(model_dict))
models = pd.DataFrame(model_score_df(model_dict))
print(models)

print('saving models performance, please wait...')
conn = psycopg2.connect(user="postgres", password="postgres",
                                  host="127.0.0.1", port="5432", database="mike")

print('deleting table from database if exist')
with conn:
    cursor = conn.cursor()
    cursor.execute('drop table if exists models_performance')
    conn.commit()

print('exporting the dataframe to postgres table...')
engine = create_engine('postgresql://postgres:postgres@localhost:5432/mike')
models.to_sql('models_performance', engine, chunksize=1000)
print('data loaded to postgres successfully...') 

#Define the best models with the selected params from the grdsearch
lr_best_model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, 
                                    C=1.0, fit_intercept=True, intercept_scaling=1,
                                    class_weight=None, random_state=None, 
                                    solver='lbfgs', max_iter=1000, multi_class='auto',
                                    verbose=0, warm_start=False, n_jobs=None, 
                                    l1_ratio=None)

ann_best_model = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                               beta_2=0.999, early_stopping=False, epsilon=1e-08,
                               hidden_layer_sizes=(50, 50, 50), learning_rate='constant',
                               learning_rate_init=0.001, max_iter=200, momentum=0.9,
                               nesterovs_momentum=True, power_t=0.5, random_state=None,
                               shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
                               verbose=False, warm_start=False)
dt_best_model = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                                           max_features=None, max_leaf_nodes=None,
                                              min_impurity_split=1e-07, min_samples_leaf=1,
                                              min_samples_split=2, min_weight_fraction_leaf=0.0,
                                              random_state=None, splitter='best')

xgboost_best_model =  XGBClassifier(objective="binary:logistic", n_estimators=20, random_state=42, num_class=11,
                                  learning_rate=0.1,
                                  num_iterations=1000,
                                  max_depth=10,
                                  feature_fraction=0.7, 
                                  scale_pos_weight=1.5,
                                  boosting='gbdt',
                                  metric='multiclass',
                                  eval_metric='mlogloss')

nbc_best_model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

rf_best_model =RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                              class_weight=None,
                                              criterion='gini', max_depth=None,
                                              max_features='auto',
                                              max_leaf_nodes=None,
                                              max_samples=None,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              n_estimators=100, n_jobs=None,
                                              oob_score=False,
                                              random_state=None, verbose=0,
                                              warm_start=False)

svm_best_model = SVC(C=1.0, break_ties=False, cache_size=200,
                           class_weight=None, coef0=0.0,
                           decision_function_shape='ovr', degree=3,
                           gamma='scale', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False)

knn_best_model = KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                            metric='minkowski',
                                            metric_params=None, n_jobs=None,
                                            n_neighbors=5, p=2,
                                            weights='uniform')
print('printing confusion matrix for LR...')
#Confusion Matrix - SGD
#Train test split with stratified sampling. Using non-binarized labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, shuffle = True, stratify = y, random_state = 3)
#Fit the training data
lr_best_model.fit(X_train, y_train)
#Predict the testing data
y_pred = lr_best_model.predict(X_test)
#Get the confusion matrix and put it into a df
cm = confusion_matrix(y_test, y_pred) 
cm_df = pd.DataFrame(cm, index = ['HIV/AIDS & TB','Other infectious', 'Metabolic',
                                  'Cardiovascular', 'Indeterminate', 'Martenal & neonatal', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'], 
                     columns = ['HIV/AIDS & TB','Indeterminate','Other infectious', 'Metabolic',
                                  'Cardiovascular', 'Martenal & neonatal', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'])
#Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cm_df, center=0, cmap=sns.diverging_palette(220, 15, as_cmap=True), annot=True, fmt='g')
plt.title('LR (loss = log) \nF1 Score (avg = macro) : {0:.2f}'.format(f1_score(y_test, y_pred, average='macro')), fontsize = 13)
plt.ylabel('True label', fontsize = 13)
plt.xlabel('Predicted label', fontsize = 13)
plt.savefig('cm_lr.png', dpi = 900)
plt.savefig('cm_lr.pdf', dpi = 900)
plt.show()

print('printing roc curve for LR...')
#Plot AUC - SGD
#Binarize the labels
y_b = label_binarize(y, classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
n_classes = y_b.shape[1]
#Shuffle and split training and test sets with stratified sampling and binarized labels
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X,
                                                            y_b,
                                                            test_size = .3,
                                                            shuffle = True,
                                                            stratify = y,
                                                            random_state = 3)

lr_classifier = OneVsRestClassifier(lr_best_model)
y_score = lr_classifier.fit(X_train_b, y_train_b).predict_proba(X_test_b)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#Compute micro-average ROC curve and ROC area
fpr['micro'], tpr['micro'], _ = roc_curve(y_test_b.ravel(), y_score.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
#First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#Finally average it and compute AUC
mean_tpr /= n_classes
fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
#Plot all ROC curves
plt.figure(figsize=(13,10)) 
sns.set_style('darkgrid')
lw=2
plt.plot(fpr['micro'], 
         tpr['micro'], 
         label='micro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['micro']),
         color='deeppink',
         linestyle=':', 
         linewidth=4)
plt.plot(fpr['macro'], 
         tpr['macro'], 
         label='macro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['macro']),
         color='navy', 
         linestyle=':', 
         linewidth=4)
colors = cycle(['#41924F', '#FFC300', '#a98ff3', '#59C7EA', '#9467bd', '#e377c2',
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], 
             color=color, 
             lw=lw, 
             label='ROC curve of class {0} (area = {1:0.3f})'''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity (False Positive Rate)', fontsize = 14)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize = 14)
plt.title('Receiver Operating Characteristic - LR', fontsize = 16)
plt.legend(loc="lower right", fontsize = 13)
plt.savefig('roc_lr.png', dpi = 900)
plt.savefig('roc_lr.pdf', dpi = 900)
plt.show()

###################################################################################################

print('printing confusion matrix for ANN...')
#Confusion Matrix - SGD
#Train test split with stratified sampling. Using non-binarized labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, shuffle = True, stratify = y, random_state = 3)
#Fit the training data
ann_best_model.fit(X_train, y_train)
#Predict the testing data
y_pred = ann_best_model.predict(X_test)
#Get the confusion matrix and put it into a df
cm = confusion_matrix(y_test, y_pred) 
cm_df = pd.DataFrame(cm, index = ['HIV/AIDS & TB','Other infectious', 'Metabolic',
                                  'Cardiovascular', 'Indeterminate', 'Martenal & neonatal', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'], 
                     columns = ['HIV/AIDS & TB','Indeterminate','Other infectious', 'Metabolic',
                                  'Cardiovascular', 'Martenal & neonatal', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'])
#Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cm_df, center=0, cmap=sns.diverging_palette(220, 15, as_cmap=True), annot=True, fmt='g')
plt.title('ANN (loss = log) \nF1 Score (avg = macro) : {0:.2f}'.format(f1_score(y_test, y_pred, average='macro')), fontsize = 13)
plt.ylabel('True label', fontsize = 13)
plt.xlabel('Predicted label', fontsize = 13)
plt.savefig('cm_ann.png', dpi = 900)
plt.savefig('cm_ann.pdf', dpi = 900)
plt.show()

print('printing roc curve for ANN...')
#Plot AUC - RF
#Learn to predict each class against the other
ann_classifier = OneVsRestClassifier(ann_best_model)
y_score = ann_classifier.fit(X_train_b, y_train_b).predict_proba(X_test_b)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#Compute micro-average ROC curve and ROC area
fpr['micro'], tpr['micro'], _ = roc_curve(y_test_b.ravel(), y_score.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
#First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#Finally average it and compute AUC
mean_tpr /= n_classes
fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
#Plot all ROC curves
plt.figure(figsize=(13,10)) 
sns.set_style('darkgrid')
lw=2
plt.plot(fpr['micro'], 
         tpr['micro'], 
         label='micro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['micro']),
         color='deeppink',
         linestyle=':', 
         linewidth=4)
plt.plot(fpr['macro'], 
         tpr['macro'], 
         label='macro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['macro']),
         color='navy', 
         linestyle=':', 
         linewidth=4)
colors = cycle(['#41924F', '#FFC300', '#a98ff3', '#59C7EA', '#9467bd', '#e377c2',
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], 
             color=color, 
             lw=lw, 
             label='ROC curve of class {0} (area = {1:0.3f})'''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity (False Positive Rate)', fontsize = 14)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize = 14)
plt.title('Receiver Operating Characteristic - ANN', fontsize = 16)
plt.legend(loc="lower right", fontsize = 13)
plt.savefig('roc_ann.png', dpi = 900)
plt.savefig('roc_ann.pdf', dpi = 900)
plt.show()

################################################################################################
print('printing confusion matrix for DT...')
#Confusion Matrix - SGD
#Train test split with stratified sampling. Using non-binarized labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, shuffle = True, stratify = y, random_state = 3)
#Fit the training data
dt_best_model.fit(X_train, y_train)
#Predict the testing data
y_pred = dt_best_model.predict(X_test)
#Get the confusion matrix and put it into a df
cm = confusion_matrix(y_test, y_pred) 
cm_df = pd.DataFrame(cm, index = ['HIV/AIDS & TB','Other infectious', 'Metabolic',
                                  'Cardiovascular', 'Indeterminate', 'Martenal & neonatal', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'], 
                     columns = ['HIV/AIDS & TB','Indeterminate','Other infectious', 'Metabolic',
                                  'Cardiovascular', 'Martenal & neonatal', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'])
#Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cm_df, center=0, cmap=sns.diverging_palette(220, 15, as_cmap=True), annot=True, fmt='g')
plt.title('DT (loss = log) \nF1 Score (avg = macro) : {0:.2f}'.format(f1_score(y_test, y_pred, average='macro')), fontsize = 13)
plt.ylabel('True label', fontsize = 13)
plt.xlabel('Predicted label', fontsize = 13)
plt.savefig('cm_dt.png', dpi = 900)
plt.savefig('cm_dt.pdf', dpi = 900)
plt.show()

print('printing roc curve for DT...')
#Plot AUC - RF
#Learn to predict each class against the other
dt_classifier = OneVsRestClassifier(dt_best_model)
y_score = dt_classifier.fit(X_train_b, y_train_b).predict_proba(X_test_b)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#Compute micro-average ROC curve and ROC area
fpr['micro'], tpr['micro'], _ = roc_curve(y_test_b.ravel(), y_score.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
#First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#Finally average it and compute AUC
mean_tpr /= n_classes
fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
#Plot all ROC curves
plt.figure(figsize=(13,10)) 
sns.set_style('darkgrid')
lw=2
plt.plot(fpr['micro'], 
         tpr['micro'], 
         label='micro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['micro']),
         color='deeppink',
         linestyle=':', 
         linewidth=4)
plt.plot(fpr['macro'], 
         tpr['macro'], 
         label='macro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['macro']),
         color='navy', 
         linestyle=':', 
         linewidth=4)
colors = cycle(['#41924F', '#FFC300', '#a98ff3', '#59C7EA', '#9467bd', '#e377c2',
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], 
             color=color, 
             lw=lw, 
             label='ROC curve of class {0} (area = {1:0.3f})'''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity (False Positive Rate)', fontsize = 14)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize = 14)
plt.title('Receiver Operating Characteristic - DT', fontsize = 16)
plt.legend(loc="lower right", fontsize = 13)
plt.savefig('roc_dt.png', dpi = 900)
plt.savefig('roc_dt.pdf', dpi = 900)
plt.show()

####################################################################################################
print('printing confusion matrix for XGBoost...')
#Confusion Matrix - SGD
#Train test split with stratified sampling. Using non-binarized labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, shuffle = True, stratify = y, random_state = 3)
#Fit the training data
xgboost_best_model.fit(X_train, y_train)
#Predict the testing data
y_pred = xgboost_best_model.predict(X_test)
#Get the confusion matrix and put it into a df
cm = confusion_matrix(y_test, y_pred) 
cm_df = pd.DataFrame(cm, index = ['HIV/AIDS & TB','Other infectious', 'Metabolic',
                                  'Cardiovascular', 'Indeterminate', 'Martenal & neonatal', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'], 
                     columns = ['HIV/AIDS & TB','Indeterminate','Other infectious', 'Metabolic',
                                  'Cardiovascular', 'Martenal & neonatal', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'])
#Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cm_df, center=0, cmap=sns.diverging_palette(220, 15, as_cmap=True), annot=True, fmt='g')
plt.title('XGBoost (loss = log) \nF1 Score (avg = macro) : {0:.2f}'.format(f1_score(y_test, y_pred, average='macro')), fontsize = 13)
plt.ylabel('True label', fontsize = 13)
plt.xlabel('Predicted label', fontsize = 13)
plt.savefig('cm_xgb.png', dpi = 900)
plt.savefig('cm_xgb.pdf', dpi = 900)
plt.show()

print('printing roc curve for XGBoost...')
#Plot AUC - RF
#Learn to predict each class against the other
xgb_classifier = OneVsRestClassifier(xgboost_best_model)
y_score = xgb_classifier.fit(X_train_b, y_train_b).predict_proba(X_test_b)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#Compute micro-average ROC curve and ROC area
fpr['micro'], tpr['micro'], _ = roc_curve(y_test_b.ravel(), y_score.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
#First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#Finally average it and compute AUC
mean_tpr /= n_classes
fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
#Plot all ROC curves
plt.figure(figsize=(13,10)) 
sns.set_style('darkgrid')
lw=2
plt.plot(fpr['micro'], 
         tpr['micro'], 
         label='micro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['micro']),
         color='deeppink',
         linestyle=':', 
         linewidth=4)
plt.plot(fpr['macro'], 
         tpr['macro'], 
         label='macro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['macro']),
         color='navy', 
         linestyle=':', 
         linewidth=4)
colors = cycle(['#41924F', '#FFC300', '#a98ff3', '#59C7EA', '#9467bd', '#e377c2',
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], 
             color=color, 
             lw=lw, 
             label='ROC curve of class {0} (area = {1:0.3f})'''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity (False Positive Rate)', fontsize = 14)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize = 14)
plt.title('Receiver Operating Characteristic - XGBoost', fontsize = 16)
plt.legend(loc="lower right", fontsize = 13)
plt.savefig('roc_xgb.png', dpi = 900)
plt.savefig('roc_xgb.pdf', dpi = 900)
plt.show()

####################################################################################################

print('printing confusion matrix for NBC...')
#Confusion Matrix - SGD
#Train test split with stratified sampling. Using non-binarized labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, shuffle = True, stratify = y, random_state = 3)
#Fit the training data
nbc_best_model.fit(X_train, y_train)
#Predict the testing data
y_pred = nbc_best_model.predict(X_test)
#Get the confusion matrix and put it into a df
cm = confusion_matrix(y_test, y_pred) 
cm_df = pd.DataFrame(cm, index = ['HIV/AIDS & TB','Other infectious', 'Metabolic',
                                  'Cardiovascular', 'Indeterminate', 'Martenal & neonatal', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'], 
                     columns = ['HIV/AIDS & TB','Indeterminate','Other infectious', 'Metabolic',
                                  'Cardiovascular', 'Martenal & neonatal', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'])
#Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cm_df, center=0, cmap=sns.diverging_palette(220, 15, as_cmap=True), annot=True, fmt='g')
plt.title('NBC (loss = log) \nF1 Score (avg = macro) : {0:.2f}'.format(f1_score(y_test, y_pred, average='macro')), fontsize = 13)
plt.ylabel('True label', fontsize = 13)
plt.xlabel('Predicted label', fontsize = 13)
plt.savefig('cm_nbc.png', dpi = 900)
plt.savefig('cm_nbc.pdf', dpi = 900)
plt.show()

print('printing roc curve for NBC...')
#Plot AUC - RF
#Learn to predict each class against the other
nbc_classifier = OneVsRestClassifier(nbc_best_model)
y_score = nbc_classifier.fit(X_train_b, y_train_b).predict_proba(X_test_b)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#Compute micro-average ROC curve and ROC area
fpr['micro'], tpr['micro'], _ = roc_curve(y_test_b.ravel(), y_score.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
#First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#Finally average it and compute AUC
mean_tpr /= n_classes
fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
#Plot all ROC curves
plt.figure(figsize=(13,10)) 
sns.set_style('darkgrid')
lw=2
plt.plot(fpr['micro'], 
         tpr['micro'], 
         label='micro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['micro']),
         color='deeppink',
         linestyle=':', 
         linewidth=4)
plt.plot(fpr['macro'], 
         tpr['macro'], 
         label='macro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['macro']),
         color='navy', 
         linestyle=':', 
         linewidth=4)
colors = cycle(['#41924F', '#FFC300', '#a98ff3', '#59C7EA', '#9467bd', '#e377c2',
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], 
             color=color, 
             lw=lw, 
             label='ROC curve of class {0} (area = {1:0.3f})'''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity (False Positive Rate)', fontsize = 14)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize = 14)
plt.title('Receiver Operating Characteristic - NBC', fontsize = 16)
plt.legend(loc="lower right", fontsize = 13)
plt.savefig('roc_nbc.png', dpi = 900)
plt.savefig('roc_nbc.pdf', dpi = 900)
plt.show()

##################################################################################################
print('printing confusion matrix for KNN...')
#Confusion Matrix - KNN
#Train test split with stratified sampling. Using non-binarized labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, shuffle = True, stratify = y, random_state = 3)
#Fit the training data
knn_best_model.fit(X_train, y_train)
#Predict the testing data
y_pred = knn_best_model.predict(X_test)
#Get the confusion matrix and put it into a df
cm = confusion_matrix(y_test, y_pred) 
cm_df = pd.DataFrame(cm, index = ['HIV/AIDS & TB','Other infectious','Indeterminate', 'Metabolic',
                                  'Cardiovascular', 'Martenal & neonatal', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'], 
                     columns = ['HIV/AIDS & TB','Other infectious','Indeterminate', 'Metabolic',
                                  'Cardiovascular', 'Martenal & neonatal', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'])
#Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cm_df, center=0, cmap=sns.diverging_palette(220, 15, as_cmap=True), annot=True, fmt='g')
plt.title('KNN \nF1 Score (avg = macro) : {0:.2f}'.format(f1_score(y_test, y_pred, average='macro')), fontsize = 20)
plt.ylabel('True label', fontsize = 18)
plt.xlabel('Predicted label', fontsize = 18)
plt.savefig('cm_knn.png', dpi = 900)
plt.savefig('cm_knn.pdf', dpi = 900)
plt.show()

print('printing roc curve for KNN...')
#Plot AUC - RF
#Learn to predict each class against the other
knn_classifier = OneVsRestClassifier(knn_best_model)
y_score = knn_classifier.fit(X_train_b, y_train_b).predict_proba(X_test_b)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#Compute micro-average ROC curve and ROC area
fpr['micro'], tpr['micro'], _ = roc_curve(y_test_b.ravel(), y_score.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
#First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#Finally average it and compute AUC
mean_tpr /= n_classes
fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
#Plot all ROC curves
plt.figure(figsize=(13,10)) 
sns.set_style('darkgrid')
lw=2
plt.plot(fpr['micro'], 
         tpr['micro'], 
         label='micro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['micro']),
         color='deeppink',
         linestyle=':', 
         linewidth=4)
plt.plot(fpr['macro'], 
         tpr['macro'], 
         label='macro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['macro']),
         color='navy', 
         linestyle=':', 
         linewidth=4)
colors = cycle(['#41924F', '#FFC300', '#a98ff3', '#59C7EA', '#9467bd', '#e377c2',
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], 
             color=color, 
             lw=lw, 
             label='ROC curve of class {0} (area = {1:0.3f})'''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity (False Positive Rate)', fontsize = 14)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize = 14)
plt.title('Receiver Operating Characteristic - KNN', fontsize = 16)
plt.legend(loc="lower right", fontsize = 13)
plt.savefig('roc_knn.png', dpi = 900)
plt.savefig('roc_knn.pdf', dpi = 900)
plt.show()

##################################################################################################


print('printing confusion matrix for SVM...')
#Confusion Matrix - SVM
#Train test split with stratified sampling. Using non-binarized labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, shuffle = True, stratify = y, random_state = 3)
#Fit the training data
svm_best_model.fit(X_train, y_train)
#Predict the testing data
y_pred = svm_best_model.predict(X_test)
#Get the confusion matrix and put it into a df
cm = confusion_matrix(y_test, y_pred) 
cm_df = pd.DataFrame(cm, index = ['HIV/AIDS & TB','Other infectious', 'Metabolic',
                                  'Cardiovascular', 'Martenal & neonatal','Indeterminate', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'], 
                     columns = ['HIV/AIDS & TB','Other infectious', 'Metabolic','Indeterminate',
                                  'Cardiovascular', 'Martenal & neonatal', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'])
#Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cm_df, center=0, cmap=sns.diverging_palette(220, 15, as_cmap=True), annot=True, fmt='g')
plt.title('SVM  \nF1 Score (avg = macro) : {0:.2f}'.format(f1_score(y_test, y_pred, average='macro')), fontsize = 20)
plt.ylabel('True label', fontsize = 18)
plt.xlabel('Predicted label', fontsize = 18)
plt.savefig('cm_svm.png', dpi = 900)
plt.savefig('cm_svm.pdf', dpi = 900)
plt.show()

print('printing roc curve for SVM...')
#Plot AUC - RF
#Learn to predict each class against the other
svm_classifier = OneVsRestClassifier(svm_best_model)
y_score = svm_classifier.fit(X_train_b, y_train_b).predict_proba(X_test_b)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#Compute micro-average ROC curve and ROC area
fpr['micro'], tpr['micro'], _ = roc_curve(y_test_b.ravel(), y_score.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
#First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#Finally average it and compute AUC
mean_tpr /= n_classes
fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
#Plot all ROC curves
plt.figure(figsize=(13,10)) 
sns.set_style('darkgrid')
lw=2
plt.plot(fpr['micro'], 
         tpr['micro'], 
         label='micro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['micro']),
         color='deeppink',
         linestyle=':', 
         linewidth=4)
plt.plot(fpr['macro'], 
         tpr['macro'], 
         label='macro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['macro']),
         color='navy', 
         linestyle=':', 
         linewidth=4)
colors = cycle(['#41924F', '#FFC300', '#a98ff3', '#59C7EA', '#9467bd', '#e377c2',
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], 
             color=color, 
             lw=lw, 
             label='ROC curve of class {0} (area = {1:0.3f})'''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity (False Positive Rate)', fontsize = 14)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize = 14)
plt.title('Receiver Operating Characteristic - SVM', fontsize = 16)
plt.legend(loc="lower right", fontsize = 13)
plt.savefig('roc_svm.png', dpi = 900)
plt.savefig('roc_svm.pdf', dpi = 900)
plt.show()



##############################################################################################



print('printing confusion matrix for RF...')
#Confusion Matrix - RF
#Fit the training data
rf_best_model.fit(X_train, y_train)
#Predict the testing data
y_pred = rf_best_model.predict(X_test)
#Get the confusion matrix and put it into a df
cm = confusion_matrix(y_test, y_pred) 
cm_df = pd.DataFrame(cm, index = ['HIV/AIDS & TB','Other infectious', 'Indeterminate','Metabolic',
                                  'Cardiovascular', 'Martenal & neonatal', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'], 
                     columns = ['HIV/AIDS & TB','Other infectious', 'Indeterminate','Metabolic',
                                  'Cardiovascular', 'Martenal & neonatal', 'Abdominal',
                                  'Neoplasms', 'External causes', 'Neurological', 
                                  'Respiratory', 'Other NCD'])
#Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cm_df, center=0, cmap=sns.diverging_palette(220, 15, as_cmap=True), annot=True, fmt='g')
plt.title('Random Forest \nF1 Score (avg = macro) : {0:.2f}'.format(f1_score(y_test, y_pred, average='macro')), fontsize = 20)
plt.ylabel('True label', fontsize = 18)
plt.xlabel('Predicted label', fontsize = 18)
plt.tight_layout()
plt.savefig('cm_rf.png', dpi = 900)
plt.savefig('cm_rf.pdf', dpi = 900)
plt.show()

print('printing roc curve for RF...')
#Plot AUC - RF
#Learn to predict each class against the other
rf_classifier = OneVsRestClassifier(rf_best_model)
y_score = rf_classifier.fit(X_train_b, y_train_b).predict_proba(X_test_b)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#Compute micro-average ROC curve and ROC area
fpr['micro'], tpr['micro'], _ = roc_curve(y_test_b.ravel(), y_score.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
#First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#Finally average it and compute AUC
mean_tpr /= n_classes
fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
#Plot all ROC curves
plt.figure(figsize=(13,10)) 
sns.set_style('darkgrid')
lw=2
plt.plot(fpr['micro'], 
         tpr['micro'], 
         label='micro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['micro']),
         color='deeppink',
         linestyle=':', 
         linewidth=4)
plt.plot(fpr['macro'], 
         tpr['macro'], 
         label='macro-average ROC curve (area = {0:0.3f})'''.format(roc_auc['macro']),
         color='navy', 
         linestyle=':', 
         linewidth=4)
colors = cycle(['#41924F', '#FFC300', '#a98ff3', '#59C7EA', '#9467bd', '#e377c2',
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], 
             color=color, 
             lw=lw, 
             label='ROC curve of class {0} (area = {1:0.3f})'''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity (False Positive Rate)', fontsize = 14)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize = 14)
plt.title('Receiver Operating Characteristic - RF', fontsize = 16)
plt.legend(loc="lower right", fontsize = 13)
plt.savefig('roc_rf.png', dpi = 900)
plt.savefig('roc_rf.pdf', dpi = 900)
plt.show()

##############################################################################################


print('performing actual classification, please wait')
#Use the saved models to transform the unseen text with tf-idf and lsa
X_unseen_x1 = dfA[dfA.columns[~dfA.columns.isin(['icdmain_consensus','broad_cause_cat', 
                                                      'label_num','id', 'disease_description'])]]
X_unseen_x2 = tfidf_vectorizer.transform(dfA['disease_description'].values.astype(str)) 

X_unseen_x = hstack([X_unseen_x1, X_unseen_x2])

X_unseen = lsa.transform(X_unseen_x)


#Fit the models with the best params on the full data
lr_best_model.fit(X, y)
rf_best_model.fit(X, y)
svm_best_model.fit(X, y)
knn_best_model.fit(X, y)
ann_best_model.fit(X, y)
dt_best_model.fit(X, y)
xgboost_best_model.fit(X, y)
nbc_best_model.fit(X, y)

#Make the prediction on the unseen articles with the fitted best models and put it into a df alongside the correct labels
dfA['pred_lr'] = lr_best_model.predict(X_unseen)
dfA['pred_rf'] = rf_best_model.predict(X_unseen)
dfA['pred_svm'] = svm_best_model.predict(X_unseen)
dfA['pred_knn'] = knn_best_model.predict(X_unseen)
dfA['pred_xgboost'] = xgboost_best_model.predict(X_unseen)
dfA['pred_ann'] = ann_best_model.predict(X_unseen)
dfA['pred_dt'] = dt_best_model.predict(X_unseen)
dfA['pred_nbc'] = nbc_best_model.predict(X_unseen)


print('reset the index...','\n')
#dfA.set_index(['id'], inplace=True)
print(len(dfA))
print(dfA.head())

conn = psycopg2.connect(user="postgres", password="postgres",
                                  host="127.0.0.1", port="5432", database="mike")

print('deleting table from database if exist')
with conn:
    cursor = conn.cursor()
    cursor.execute('drop table if exists ml_classified_cases2')
    conn.commit()

print('exporting the dataframe to postgres table...')
engine = create_engine('postgresql://postgres:postgres@localhost:5432/mike')
dfA.to_sql('ml_classified_case2', engine, chunksize=1000, index=False)
print('data loaded to postgres successfully...') 

print('halting...') 
stoptd = time.strftime('%X %x %Z')
print('\n','The program stop time and Date','\n',stoptd)
print('It took', (time.time()-start)/60, 'minutes to run the script.')


print('self clean up...')
del dfA, dfA_subA, conn, cursor, sql_cmd1, start, startptd, stoptd
del X, X_test, X_train, X_unseen, X_unseen_tfidf, engine, knn_best_model, y_pred
del lsa, model_dict, models, rf_best_model, nbc_best_model, lr_best_model, xbg_best_model, ann_best_model, dt_best_model, svm_best_model
del y_test, y_train, strategy, df, sql_cmd2, df1, df2, y, tfidf_vectorizer
del dfA_subB, cm, cm_df, oversample
print('complete...')  
