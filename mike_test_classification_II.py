# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:06:50 2021

@author: VictorO
"""

print('importing required packages...')
import pandas as pd, numpy as np, os, re, time, matplotlib.pyplot as plt, seaborn as sns, warnings, psycopg2
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
#from imblearn.over_sampling import ROSE
from imblearn.combine import SMOTETomek
import statsmodels.api as sm
import xgboost as xgb
from scipy.sparse import hstack 
from scipy import interp
from itertools import cycle
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from imblearn.over_sampling import RandomOverSampler

#from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier

#from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, label_binarize
'''Metrics/Evaluation'''
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

'''Display'''
from IPython.core.display import display, HTML
import nltk
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
os.chdir('C:\\Users\\a0056407\\Desktop\\Michael_Mapundu__Docs\\PhDPythonScript\\FinalMLCombinedVAS\\')
#df = pd.read_csv('C:\\Users\\a0056407\\Desktop\\Michael_Mapundu__Docs\\PhDPythonScript\\FinalMLCombinedVAS\\MichaelData.csv', encoding='latin1')
print('reading the dataset, please wait...')

#Converting the columns headers to lower case
#df.columns = df.columns.str.lower()

conn = psycopg2.connect(user="postgres", password="postgres",
                                  host="127.0.0.1", port="5432", database="mike")
print('deleting table from database if exist')
with conn:
    cursor = conn.cursor()
    cursor.execute('drop table if exists encryted_ncr')
    conn.commit()

print('exporting the dataframe to postgres table...')
engine = create_engine('postgresql://postgres:postgres@localhost:5432/mike')
#df.to_sql('verbal_autopsy', engine, chunksize=10000, index=False)
print('data loaded to postgres successfully...')
cursor = conn.cursor()
sql_cmd1 = "select * from verbal_autopsy"
cursor.execute(sql_cmd1)
#cursor.execute(sql_cmd2)
dfA = pd.read_sql(sql_cmd1, conn)
#df2 = pd.read_sql(sql_cmd2, conn)
#closing the connection
cursor.close()
conn.close()
print('data read successfully and connection closed...')

for col in dfA.columns:
    print(col)

#drop less important rows
del dfA['dob'], dfA['dod'], dfA['elder'], dfA['midage'], dfA['adult'], dfA['child'], dfA['province'], dfA['womandeath']
del dfA['under5'], dfA['infant'], dfA['neonate'], dfA['deathlocal'], dfA['diedat'], dfA['diedatplace'], dfA['hospital']
del dfA['deathregistration'], dfA['respondentpresent'], dfA['whynorespondent'], dfA['respondentdeceasedrelation']
del dfA['otherrelation'], dfA['famcause1'], dfA['famcause2'], dfA['receiv_biotreatment']
del dfA['biomedi_received'], dfA['receiv_tradi_treatment'], dfA['tradimed_received'], dfA['treatm_sougth_first'], dfA['other_remarks']
del dfA['icdimed_consensus'], dfA['icdcont_consensus'], dfA['maincod_assess1'], dfA['icdmain_assess1'], dfA['lik1'], dfA['lik3']
del dfA['immcod_assess1'], dfA['contrcod_assess1'], dfA['doctor_name_assess1'], dfA['maincod_assess2'], dfA['lik2'], dfA['cause3'] 
del dfA['icdmain_assess2'], dfA['immcod_assess2'], dfA['contrcod_assess2'], dfA['doctor_name_assess2'], dfA['cause2']
del dfA['malprev'], dfA['hivprev'], dfA['pregstat'], dfA['preglik'], dfA['prmat'], dfA['indet'], dfA['cause1']

#recoding data
dfA['death_age'] = dfA['death_age'].fillna(0)
dfA['death_age'] = pd.to_numeric(dfA['death_age'], errors='coerce')
dfA['male'] = np.where(dfA['male']=='y', '1', '0')
dfA['male'] = pd.to_numeric(dfA['male'], errors='coerce')
dfA['female'] = np.where(dfA['female']=='y', '1', '0')
dfA['female'] = pd.to_numeric(dfA['female'], errors='coerce')
dfA['acute'] = np.where(dfA['acute']=='y', '1', '0')
dfA['acute'] = pd.to_numeric(dfA['acute'], errors='coerce')
dfA['chronic'] = np.where(dfA['chronic']=='y', '1', '0')
dfA['chronic'] = pd.to_numeric(dfA['chronic'], errors='coerce')
dfA['sudden'] = np.where(dfA['female']=='y', '1', '0')
dfA['sudden'] = pd.to_numeric(dfA['sudden'], errors='coerce')
dfA['wet_seas'] = np.where(dfA['wet_seas']=='y', '1', '0')
dfA['wet_seas'] = pd.to_numeric(dfA['wet_seas'], errors='coerce')
dfA['dry_seas'] = np.where(dfA['dry_seas']=='y', '1', '0')
dfA['dry_seas'] = pd.to_numeric(dfA['dry_seas'], errors='coerce')
dfA['heart_dis'] = np.where(dfA['heart_dis']=='y', '1', '0')
dfA['heart_dis'] = pd.to_numeric(dfA['heart_dis'], errors='coerce')
dfA['tuber'] = np.where(dfA['tuber']=='y', '1', '0')
dfA['tuber'] = pd.to_numeric(dfA['tuber'], errors='coerce')
dfA['hiv_aids'] = np.where(dfA['hiv_aids']=='y', '1', '0')
dfA['hiv_aids'] = pd.to_numeric(dfA['hiv_aids'], errors='coerce')
dfA['hypert'] = np.where(dfA['hypert']=='y', '1', '0')
dfA['hypert'] = pd.to_numeric(dfA['hypert'], errors='coerce')
dfA['diabetes'] = np.where(dfA['diabetes']=='y', '1', '0')
dfA['diabetes'] = pd.to_numeric(dfA['diabetes'], errors='coerce')
dfA['asthma'] = np.where(dfA['asthma']=='y', '1', '0')
dfA['asthma'] = pd.to_numeric(dfA['asthma'], errors='coerce')
dfA['epilepsy'] = np.where(dfA['epilepsy']=='y', '1', '0')
dfA['epilepsy'] = pd.to_numeric(dfA['epilepsy'], errors='coerce')
dfA['cancer'] = np.where(dfA['cancer']=='y', '1', '0')
dfA['cancer'] = pd.to_numeric(dfA['cancer'], errors='coerce')
dfA['copd'] = np.where(dfA['copd']=='y', '1', '0')
dfA['copd'] = pd.to_numeric(dfA['copd'], errors='coerce')
dfA['dement'] = np.where(dfA['dement']=='y', '1', '0')
dfA['dement'] = pd.to_numeric(dfA['dement'], errors='coerce')
dfA['depress'] = np.where(dfA['depress']=='y', '1', '0')
dfA['depress'] = pd.to_numeric(dfA['depress'], errors='coerce')
dfA['stroke'] = np.where(dfA['stroke']=='y', '1', '0')
dfA['stroke'] = pd.to_numeric(dfA['stroke'], errors='coerce')
dfA['sickle'] = np.where(dfA['sickle']=='y', '1', '0')
dfA['sickle'] = pd.to_numeric(dfA['sickle'], errors='coerce')
dfA['kidney_dis'] = np.where(dfA['kidney_dis']=='y', '1', '0')
dfA['kidney_dis'] = pd.to_numeric(dfA['kidney_dis'], errors='coerce')
dfA['liver_dis'] = np.where(dfA['liver_dis']=='y', '1', '0')
dfA['liver_dis'] = pd.to_numeric(dfA['liver_dis'], errors='coerce')
dfA['measles'] = np.where(dfA['measles']=='y', '1', '0')
dfA['measles'] = pd.to_numeric(dfA['measles'], errors='coerce')
dfA['men_con'] = np.where(dfA['men_con']=='y', '1', '0')
dfA['men_con'] = pd.to_numeric(dfA['men_con'], errors='coerce')
dfA['mencon3'] = np.where(dfA['mencon3']=='y', '1', '0')
dfA['mencon3'] = pd.to_numeric(dfA['mencon3'], errors='coerce')
dfA['malaria'] = np.where(dfA['malaria']=='y', '1', '0')
dfA['malaria'] = pd.to_numeric(dfA['malaria'], errors='coerce')
dfA['malarneg'] = np.where(dfA['malarneg']=='y', '1', '0')
dfA['malarneg'] = pd.to_numeric(dfA['malarneg'], errors='coerce')
dfA['fever'] = np.where(dfA['fever']=='y', '1', '0')
dfA['fever'] = pd.to_numeric(dfA['fever'], errors='coerce')
dfA['ac_fever'] = np.where(dfA['ac_fever']=='y', '1', '0')
dfA['ac_fever'] = pd.to_numeric(dfA['ac_fever'], errors='coerce')
dfA['ch_fever'] = np.where(dfA['ch_fever']=='y', '1', '0')
dfA['ch_fever'] = pd.to_numeric(dfA['ch_fever'], errors='coerce')
dfA['night_sw'] = np.where(dfA['night_sw']=='y', '1', '0')
dfA['night_sw'] = pd.to_numeric(dfA['night_sw'], errors='coerce')
dfA['cough'] = np.where(dfA['cough']=='y', '1', '0')
dfA['cough'] = pd.to_numeric(dfA['cough'], errors='coerce')
dfA['ac_cough'] = np.where(dfA['ac_cough']=='y', '1', '0')
dfA['ac_cough'] = pd.to_numeric(dfA['ac_cough'], errors='coerce')
dfA['ch_cough'] = np.where(dfA['ch_cough']=='y', '1', '0')
dfA['ch_cough'] = pd.to_numeric(dfA['ch_cough'], errors='coerce')
dfA['pr_cough'] = np.where(dfA['pr_cough']=='y', '1', '0')
dfA['pr_cough'] = pd.to_numeric(dfA['pr_cough'], errors='coerce')
dfA['bl_cough'] = np.where(dfA['bl_cough']=='y', '1', '0')
dfA['bl_cough'] = pd.to_numeric(dfA['bl_cough'], errors='coerce')
dfA['whoop'] = np.where(dfA['whoop']=='y', '1', '0')
dfA['whoop'] = pd.to_numeric(dfA['whoop'], errors='coerce')
dfA['breath'] = np.where(dfA['breath']=='y', '1', '0')
dfA['breath'] = pd.to_numeric(dfA['breath'], errors='coerce')
dfA['rapid_br'] = np.where(dfA['rapid_br']=='y', '1', '0')
dfA['rapid_br'] = pd.to_numeric(dfA['rapid_br'], errors='coerce')
dfA['ac_rpbr'] = np.where(dfA['ac_rpbr']=='y', '1', '0')
dfA['ac_rpbr'] = pd.to_numeric(dfA['ac_rpbr'], errors='coerce')
dfA['ch_rpbr'] = np.where(dfA['ch_rpbr']=='y', '1', '0')
dfA['ch_rpbr'] = pd.to_numeric(dfA['ch_rpbr'], errors='coerce')
dfA['br_less'] = np.where(dfA['br_less']=='y', '1', '0')
dfA['br_less'] = pd.to_numeric(dfA['br_less'], errors='coerce')
dfA['ac_brl'] = np.where(dfA['ac_brl']=='y', '1', '0')
dfA['ac_brl'] = pd.to_numeric(dfA['ac_brl'], errors='coerce')
dfA['ch_brl'] = np.where(dfA['ch_brl']=='y', '1', '0')
dfA['ch_brl'] = pd.to_numeric(dfA['ch_brl'], errors='coerce')
dfA['exert_br'] = np.where(dfA['exert_br']=='y', '1', '0')
dfA['exert_br'] = pd.to_numeric(dfA['exert_br'], errors='coerce')
dfA['lying_br'] = np.where(dfA['lying_br']=='y', '1', '0')
dfA['lying_br'] = pd.to_numeric(dfA['lying_br'], errors='coerce')
dfA['chest_in'] = np.where(dfA['chest_in']=='y', '1', '0')
dfA['chest_in'] = pd.to_numeric(dfA['chest_in'], errors='coerce')
dfA['wheeze'] = np.where(dfA['wheeze']=='y', '1', '0')
dfA['wheeze'] = pd.to_numeric(dfA['wheeze'], errors='coerce')
dfA['ch_pain'] = np.where(dfA['ch_pain']=='y', '1', '0')
dfA['ch_pain'] = pd.to_numeric(dfA['ch_pain'], errors='coerce')
dfA['yellow'] = np.where(dfA['yellow']=='y', '1', '0')
dfA['yellow'] = pd.to_numeric(dfA['yellow'], errors='coerce')
dfA['diarr'] = np.where(dfA['diarr']=='y', '1', '0')
dfA['diarr'] = pd.to_numeric(dfA['diarr'], errors='coerce')
dfA['ac_diarr'] = np.where(dfA['ac_diarr']=='y', '1', '0')
dfA['ac_diarr'] = pd.to_numeric(dfA['ac_diarr'], errors='coerce')
dfA['pe_diarr'] = np.where(dfA['pe_diarr']=='y', '1', '0')
dfA['pe_diarr'] = pd.to_numeric(dfA['pe_diarr'], errors='coerce')
dfA['ch_diarr'] = np.where(dfA['ch_diarr']=='y', '1', '0')
dfA['ch_diarr'] = pd.to_numeric(dfA['ch_diarr'], errors='coerce')
dfA['bl_diarr'] = np.where(dfA['bl_diarr']=='y', '1', '0')
dfA['bl_diarr'] = pd.to_numeric(dfA['bl_diarr'], errors='coerce')
dfA['vomiting'] = np.where(dfA['vomiting']=='y', '1', '0')
dfA['vomiting'] = pd.to_numeric(dfA['vomiting'], errors='coerce')
dfA['bl_vomit'] = np.where(dfA['bl_vomit']=='y', '1', '0')
dfA['bl_vomit'] = pd.to_numeric(dfA['bl_vomit'], errors='coerce')
dfA['abdom'] = np.where(dfA['abdom']=='y', '1', '0')
dfA['abdom'] = pd.to_numeric(dfA['abdom'], errors='coerce')
dfA['abd_pain'] = np.where(dfA['abd_pain']=='y', '1', '0')
dfA['abd_pain'] = pd.to_numeric(dfA['abd_pain'], errors='coerce')
dfA['ac_abdp'] = np.where(dfA['ac_abdp']=='y', '1', '0')
dfA['ac_abdp'] = pd.to_numeric(dfA['ac_abdp'], errors='coerce')
dfA['ch_abdp'] = np.where(dfA['ch_abdp']=='y', '1', '0')
dfA['ch_abdp'] = pd.to_numeric(dfA['ch_abdp'], errors='coerce')
dfA['swe_abd'] = np.where(dfA['swe_abd']=='y', '1', '0')
dfA['swe_abd'] = pd.to_numeric(dfA['swe_abd'], errors='coerce')
dfA['ac_swab'] = np.where(dfA['ac_swab']=='y', '1', '0')
dfA['ac_swab'] = pd.to_numeric(dfA['ac_swab'], errors='coerce')
dfA['ch_swab'] = np.where(dfA['ch_swab']=='y', '1', '0')
dfA['ch_swab'] = pd.to_numeric(dfA['ch_swab'], errors='coerce')
dfA['abd_mass'] = np.where(dfA['abd_mass']=='y', '1', '0')
dfA['abd_mass'] = pd.to_numeric(dfA['abd_mass'], errors='coerce')
dfA['ac_abdm'] = np.where(dfA['ac_abdm']=='y', '1', '0')
dfA['ac_abdm'] = pd.to_numeric(dfA['ac_abdm'], errors='coerce')
dfA['ch_abdm'] = np.where(dfA['ch_abdm']=='y', '1', '0')
dfA['ch_abdm'] = pd.to_numeric(dfA['ch_abdm'], errors='coerce')
dfA['headache'] = np.where(dfA['headache']=='y', '1', '0')
dfA['headache'] = pd.to_numeric(dfA['headache'], errors='coerce')
dfA['skin'] = np.where(dfA['skin']=='y', '1', '0')
dfA['skin'] = pd.to_numeric(dfA['skin'], errors='coerce')
dfA['skin_les'] = np.where(dfA['skin_les']=='y', '1', '0')
dfA['skin_les'] = pd.to_numeric(dfA['skin_les'], errors='coerce')
dfA['b_norm'] = np.where(dfA['b_norm']=='y', '1', '0')
dfA['b_norm'] = pd.to_numeric(dfA['b_norm'], errors='coerce')
dfA['b_assist'] = np.where(dfA['b_assist']=='y', '1', '0')
dfA['b_assist'] = pd.to_numeric(dfA['b_assist'], errors='coerce')
dfA['b_caes'] = np.where(dfA['b_caes']=='y', '1', '0')
dfA['b_caes'] = pd.to_numeric(dfA['b_caes'], errors='coerce')
dfA['b_first'] = np.where(dfA['b_first']=='y', '1', '0')
dfA['b_first'] = pd.to_numeric(dfA['b_first'], errors='coerce')
dfA['b_more4'] = np.where(dfA['b_more4']=='y', '1', '0')
dfA['b_more4'] = pd.to_numeric(dfA['b_more4'], errors='coerce')
dfA['b_mbpr'] = np.where(dfA['b_mbpr']=='y', '1', '0')
dfA['b_mbpr'] = pd.to_numeric(dfA['b_mbpr'], errors='coerce')
dfA['b_msmds'] = np.where(dfA['b_msmds']=='y', '1', '0')
dfA['b_msmds'] = pd.to_numeric(dfA['b_msmds'], errors='coerce')
dfA['b_mcon'] = np.where(dfA['b_mcon']=='y', '1', '0')
dfA['b_mcon'] = pd.to_numeric(dfA['b_mcon'], errors='coerce')
dfA['b_mbvi'] = np.where(dfA['b_mbvi']=='y', '1', '0')
dfA['b_mbvi'] = pd.to_numeric(dfA['b_mbvi'], errors='coerce')
dfA['b_mvbl'] = np.where(dfA['b_mvbl']=='y', '1', '0')
dfA['b_mvbl'] = pd.to_numeric(dfA['b_mvbl'], errors='coerce')
dfA['b_bfac'] = np.where(dfA['b_bfac']=='y', '1', '0')
dfA['b_bfac'] = pd.to_numeric(dfA['b_bfac'], errors='coerce')
dfA['b_bhome'] = np.where(dfA['b_bhome']=='y', '1', '0')
dfA['b_bhome'] = pd.to_numeric(dfA['b_bhome'], errors='coerce')
dfA['b_bway'] = np.where(dfA['b_bway']=='y', '1', '0')
dfA['b_bway'] = pd.to_numeric(dfA['b_bway'], errors='coerce')
dfA['b_bprof'] = np.where(dfA['b_bprof']=='y', '1', '0')
dfA['b_bprof'] = pd.to_numeric(dfA['b_bprof'], errors='coerce')
dfA['injury'] = np.where(dfA['injury']=='y', '1', '0')
dfA['injury'] = pd.to_numeric(dfA['injury'], errors='coerce')
dfA['traffic'] = np.where(dfA['traffic']=='y', '1', '0')
dfA['traffic'] = pd.to_numeric(dfA['traffic'], errors='coerce')
dfA['o_trans'] = np.where(dfA['o_trans']=='y', '1', '0')
dfA['o_trans'] = pd.to_numeric(dfA['o_trans'], errors='coerce')
dfA['fall'] = np.where(dfA['fall']=='y', '1', '0')
dfA['fall'] = pd.to_numeric(dfA['fall'], errors='coerce')
dfA['fire'] = np.where(dfA['fire']=='y', '1', '0')
dfA['fire'] = pd.to_numeric(dfA['fire'], errors='coerce')
dfA['drown'] = np.where(dfA['drown']=='y', '1', '0')
dfA['drown'] = pd.to_numeric(dfA['drown'], errors='coerce')
dfA['assault'] = np.where(dfA['assault']=='y', '1', '0')
dfA['assault'] = pd.to_numeric(dfA['assault'], errors='coerce')
dfA['vemon'] = np.where(dfA['vemon']=='y', '1', '0')
dfA['vemon'] = pd.to_numeric(dfA['vemon'], errors='coerce')
dfA['force'] = np.where(dfA['force']=='y', '1', '0')
dfA['force'] = pd.to_numeric(dfA['force'], errors='coerce')
dfA['smoking'] = np.where(dfA['smoking']=='y', '1', '0')
dfA['smoking'] = pd.to_numeric(dfA['smoking'], errors='coerce')
dfA['alcohol'] = np.where(dfA['alcohol']=='y', '1', '0')
dfA['alcohol'] = pd.to_numeric(dfA['alcohol'], errors='coerce')
dfA['poison'] = np.where(dfA['poison']=='y', '1', '0')
dfA['poison'] = pd.to_numeric(dfA['poison'], errors='coerce')
dfA['inflict'] = np.where(dfA['inflict']=='y', '1', '0')
dfA['inflict'] = pd.to_numeric(dfA['inflict'], errors='coerce')
dfA['suicide'] = np.where(dfA['suicide']=='y', '1', '0')
dfA['suicide'] = pd.to_numeric(dfA['suicide'], errors='coerce')
dfA['married'] = np.where(dfA['married']=='y', '1', '0')
dfA['married'] = pd.to_numeric(dfA['married'], errors='coerce')
dfA['vaccin'] = np.where(dfA['vaccin']=='y', '1', '0')
dfA['vaccin'] = pd.to_numeric(dfA['vaccin'], errors='coerce')
dfA['disch'] = np.where(dfA['disch']=='y', '1', '0')
dfA['disch'] = pd.to_numeric(dfA['disch'], errors='coerce')
dfA['smore2'] = np.where(dfA['smore2']=='y', '1', '0')
dfA['smore2'] = pd.to_numeric(dfA['smore2'], errors='coerce')
dfA['scosts'] = np.where(dfA['scosts']=='y', '1', '0')
dfA['scosts'] = pd.to_numeric(dfA['scosts'], errors='coerce')
dfA['smobph'] = np.where(dfA['smobph']=='y', '1', '0')
dfA['smobph'] = pd.to_numeric(dfA['smobph'], errors='coerce')
dfA['stradm'] = np.where(dfA['stradm']=='y', '1', '0')
dfA['stradm'] = pd.to_numeric(dfA['stradm'], errors='coerce')
dfA['sdoubt'] = np.where(dfA['sdoubt']=='y', '1', '0')
dfA['sdoubt'] = pd.to_numeric(dfA['sdoubt'], errors='coerce')
dfA['sur_1m'] = np.where(dfA['sur_1m']=='y', '1', '0')
dfA['sur_1m'] = pd.to_numeric(dfA['sur_1m'], errors='coerce')
dfA['treat'] = np.where(dfA['treat']=='y', '1', '0')
dfA['treat'] = pd.to_numeric(dfA['treat'], errors='coerce')
dfA['streat'] = np.where(dfA['streat']=='y', '1', '0')
dfA['streat'] = pd.to_numeric(dfA['streat'], errors='coerce')
dfA['smedic'] = np.where(dfA['smedic']=='y', '1', '0')
dfA['smedic'] = pd.to_numeric(dfA['smedic'], errors='coerce')
dfA['strans'] = np.where(dfA['strans']=='y', '1', '0')
dfA['strans'] = pd.to_numeric(dfA['strans'], errors='coerce')
dfA['sadmit'] = np.where(dfA['sadmit']=='y', '1', '0')
dfA['sadmit'] = pd.to_numeric(dfA['sadmit'], errors='coerce')
dfA['shospf'] = np.where(dfA['shospf']=='y', '1', '0')
dfA['shospf'] = pd.to_numeric(dfA['shospf'], errors='coerce')
dfA['t_ort'] = np.where(dfA['t_ort']=='y', '1', '0')
dfA['t_ort'] = pd.to_numeric(dfA['t_ort'], errors='coerce')
dfA['antib_i'] = np.where(dfA['antib_i']=='y', '1', '0')
dfA['antib_i'] = pd.to_numeric(dfA['antib_i'], errors='coerce')
dfA['surgery'] = np.where(dfA['surgery']=='y', '1', '0')
dfA['surgery'] = pd.to_numeric(dfA['surgery'], errors='coerce')
dfA['blood_tr'] = np.where(dfA['blood_tr']=='y', '1', '0')
dfA['blood_tr'] = pd.to_numeric(dfA['blood_tr'], errors='coerce')
dfA['t_iv'] = np.where(dfA['t_iv']=='y', '1', '0')
dfA['t_iv'] = pd.to_numeric(dfA['t_iv'], errors='coerce')
dfA['t_ngt'] = np.where(dfA['t_ngt']=='y', '1', '0')
dfA['t_ngt'] = pd.to_numeric(dfA['t_ngt'], errors='coerce')
dfA['sk_feet'] = np.where(dfA['sk_feet']=='y', '1', '0')
dfA['sk_feet'] = pd.to_numeric(dfA['sk_feet'], errors='coerce')
dfA['rash'] = np.where(dfA['rash']=='y', '1', '0')
dfA['rash'] = pd.to_numeric(dfA['rash'], errors='coerce')
dfA['ac_rash'] = np.where(dfA['ac_rash']=='y', '1', '0')
dfA['ac_rash'] = pd.to_numeric(dfA['ac_rash'], errors='coerce')
dfA['ch_rash'] = np.where(dfA['ch_rash']=='y', '1', '0')
dfA['ch_rash'] = pd.to_numeric(dfA['ch_rash'], errors='coerce')
dfA['measrash'] = np.where(dfA['measrash']=='y', '1', '0')
dfA['measrash'] = pd.to_numeric(dfA['measrash'], errors='coerce')
dfA['mttv'] = np.where(dfA['mttv']=='y', '1', '0')
dfA['mttv'] = pd.to_numeric(dfA['mttv'], errors='coerce')
dfA['mlf_sh'] = np.where(dfA['mlf_sh']=='y', '1', '0')
dfA['mlf_sh'] = pd.to_numeric(dfA['mlf_sh'], errors='coerce')
dfA['mlf_lh'] = np.where(dfA['mlf_lh']=='y', '1', '0')
dfA['mlf_lh'] = pd.to_numeric(dfA['mlf_lh'], errors='coerce')
dfA['mlf_bk'] = np.where(dfA['mlf_bk']=='y', '1', '0')
dfA['mlf_bk'] = pd.to_numeric(dfA['mlf_bk'], errors='coerce')
dfA['born_malf'] = np.where(dfA['born_malf']=='y', '1', '0')
dfA['born_malf'] = pd.to_numeric(dfA['born_malf'], errors='coerce')
dfA['devel'] = np.where(dfA['devel']=='y', '1', '0')
dfA['devel'] = pd.to_numeric(dfA['devel'], errors='coerce')
dfA['b_yellow'] = np.where(dfA['b_yellow']=='y', '1', '0')
dfA['b_yellow'] = pd.to_numeric(dfA['b_yellow'], errors='coerce')
dfA['umbinf'] = np.where(dfA['umbinf']=='y', '1', '0')
dfA['umbinf'] = pd.to_numeric(dfA['umbinf'], errors='coerce')
dfA['cold'] = np.where(dfA['cold']=='y', '1', '0')
dfA['cold'] = pd.to_numeric(dfA['cold'], errors='coerce')
dfA['unw_d2'] = np.where(dfA['unw_d2']=='y', '1', '0')
dfA['unw_d2'] = pd.to_numeric(dfA['unw_d2'], errors='coerce')
dfA['unw_d1'] = np.where(dfA['unw_d1']=='y', '1', '0')
dfA['unw_d1'] = pd.to_numeric(dfA['unw_d1'], errors='coerce')
dfA['font_lo'] = np.where(dfA['font_lo']=='y', '1', '0')
dfA['font_lo'] = pd.to_numeric(dfA['font_lo'], errors='coerce')
dfA['font_hi'] = np.where(dfA['font_hi']=='y', '1', '0')
dfA['font_hi'] = pd.to_numeric(dfA['font_hi'], errors='coerce')
dfA['uri_haem'] = np.where(dfA['uri_haem']=='y', '1', '0')
dfA['uri_haem'] = pd.to_numeric(dfA['uri_haem'], errors='coerce')
dfA['ac_stnk'] = np.where(dfA['ac_stnk']=='y', '1', '0')
dfA['ac_stnk'] = pd.to_numeric(dfA['ac_stnk'], errors='coerce')
dfA['convul'] = np.where(dfA['convul']=='y', '1', '0')
dfA['convul'] = pd.to_numeric(dfA['convul'], errors='coerce')
dfA['herpes'] = np.where(dfA['herpes']=='y', '1', '0')
dfA['herpes'] = pd.to_numeric(dfA['herpes'], errors='coerce')
dfA['stiff_neck'] = np.where(dfA['stiff_neck']=='y', '1', '0')
dfA['stiff_neck'] = pd.to_numeric(dfA['stiff_neck'], errors='coerce')
dfA['ch_stnk'] = np.where(dfA['ch_stnk']=='y', '1', '0')
dfA['ch_stnk'] = pd.to_numeric(dfA['ch_stnk'], errors='coerce')
dfA['coma'] = np.where(dfA['coma']=='y', '1', '0')
dfA['coma'] = pd.to_numeric(dfA['coma'], errors='coerce')
dfA['co_ons'] = np.where(dfA['co_ons']=='y', '1', '0')
dfA['co_ons'] = pd.to_numeric(dfA['co_ons'], errors='coerce')
dfA['exc_urine'] = np.where(dfA['exc_urine']=='y', '1', '0')
dfA['exc_urine'] = pd.to_numeric(dfA['exc_urine'], errors='coerce')
dfA['uri_ret'] = np.where(dfA['uri_ret']=='y', '1', '0')
dfA['uri_ret'] = pd.to_numeric(dfA['uri_ret'], errors='coerce')
dfA['urine'] = np.where(dfA['urine']=='y', '1', '0')
dfA['urine'] = pd.to_numeric(dfA['urine'], errors='coerce')
dfA['ac_conv'] = np.where(dfA['ac_conv']=='y', '1', '0')
dfA['ac_conv'] = pd.to_numeric(dfA['ac_conv'], errors='coerce')
dfA['ch_conv'] = np.where(dfA['ch_conv']=='y', '1', '0')
dfA['ch_conv'] = pd.to_numeric(dfA['ch_conv'], errors='coerce')
dfA['unc_con'] = np.where(dfA['unc_con']=='y', '1', '0')
dfA['unc_con'] = pd.to_numeric(dfA['unc_con'], errors='coerce')
dfA['hyster'] = np.where(dfA['hyster']=='y', '1', '0')
dfA['hyster'] = pd.to_numeric(dfA['hyster'], errors='coerce')
dfA['bpr_preg'] = np.where(dfA['bpr_preg']=='y', '1', '0')
dfA['bpr_preg'] = pd.to_numeric(dfA['bpr_preg'], errors='coerce')
dfA['fit_preg'] = np.where(dfA['fit_preg']=='y', '1', '0')
dfA['fit_preg'] = pd.to_numeric(dfA['fit_preg'], errors='coerce')
dfA['vis_bl'] = np.where(dfA['vis_bl']=='y', '1', '0')
dfA['vis_bl'] = pd.to_numeric(dfA['vis_bl'], errors='coerce')
dfA['bleed'] = np.where(dfA['bleed']=='y', '1', '0')
dfA['bleed'] = pd.to_numeric(dfA['bleed'], errors='coerce')
dfA['e_bleed'] = np.where(dfA['e_bleed']=='y', '1', '0')
dfA['e_bleed'] = pd.to_numeric(dfA['e_bleed'], errors='coerce')
dfA['s_bleed'] = np.where(dfA['s_bleed']=='y', '1', '0')
dfA['s_bleed'] = pd.to_numeric(dfA['s_bleed'], errors='coerce')
dfA['d_bleed'] = np.where(dfA['d_bleed']=='y', '1', '0')
dfA['d_bleed'] = pd.to_numeric(dfA['d_bleed'], errors='coerce')
dfA['p_bleed'] = np.where(dfA['p_bleed']=='y', '1', '0')
dfA['p_bleed'] = pd.to_numeric(dfA['p_bleed'], errors='coerce')
dfA['placent_r'] = np.where(dfA['placent_r']=='y', '1', '0')
dfA['placent_r'] = pd.to_numeric(dfA['placent_r'], errors='coerce')
dfA['disch_sm'] = np.where(dfA['disch_sm']=='y', '1', '0')
dfA['disch_sm'] = pd.to_numeric(dfA['disch_sm'], errors='coerce')
dfA['term_att'] = np.where(dfA['term_att']=='y', '1', '0')
dfA['term_att'] = pd.to_numeric(dfA['term_att'], errors='coerce')
dfA['abort'] = np.where(dfA['abort']=='y', '1', '0')
dfA['abort'] = pd.to_numeric(dfA['abort'], errors='coerce')
dfA['arch_b'] = np.where(dfA['arch_b']=='y', '1', '0')
dfA['arch_b'] = pd.to_numeric(dfA['arch_b'], errors='coerce')
dfA['conv_d2'] = np.where(dfA['conv_d2']=='y', '1', '0')
dfA['conv_d2'] = pd.to_numeric(dfA['conv_d2'], errors='coerce')
dfA['conv_d1'] = np.where(dfA['conv_d1']=='y', '1', '0')
dfA['conv_d1'] = pd.to_numeric(dfA['conv_d1'], errors='coerce')
dfA['ab_posit'] = np.where(dfA['ab_posit']=='y', '1', '0')
dfA['ab_posit'] = pd.to_numeric(dfA['ab_posit'], errors='coerce')
dfA['st_suck'] = np.where(dfA['st_suck']=='y', '1', '0')
dfA['st_suck'] = pd.to_numeric(dfA['st_suck'], errors='coerce')
dfA['cord'] = np.where(dfA['cord']=='y', '1', '0')
dfA['cord'] = pd.to_numeric(dfA['cord'], errors='coerce')
dfA['born_early'] = np.where(dfA['born_early']=='y', '1', '0')
dfA['born_early'] = pd.to_numeric(dfA['born_early'], errors='coerce')
dfA['born_3437'] = np.where(dfA['born_3437']=='y', '1', '0')
dfA['born_3437'] = pd.to_numeric(dfA['born_3437'], errors='coerce')
dfA['born_38'] = np.where(dfA['born_38']=='y', '1', '0')
dfA['born_38'] = pd.to_numeric(dfA['born_38'], errors='coerce')
dfA['born_nobr'] = np.where(dfA['born_nobr']=='y', '1', '0')
dfA['born_nobr'] = pd.to_numeric(dfA['born_nobr'], errors='coerce')
dfA['waters'] = np.where(dfA['waters']=='y', '1', '0')
dfA['waters'] = pd.to_numeric(dfA['waters'], errors='coerce')
dfA['ab_size'] = np.where(dfA['ab_size']=='y', '1', '0')
dfA['ab_size'] = pd.to_numeric(dfA['ab_size'], errors='coerce')
dfA['baby_br'] = np.where(dfA['baby_br']=='y', '1', '0')
dfA['baby_br'] = pd.to_numeric(dfA['baby_br'], errors='coerce')
dfA['born_big'] = np.where(dfA['born_big']=='y', '1', '0')
dfA['born_big'] = pd.to_numeric(dfA['born_big'], errors='coerce')
dfA['born_small'] = np.where(dfA['born_small']=='y', '1', '0')
dfA['born_small'] = pd.to_numeric(dfA['born_small'], errors='coerce')
dfA['fed_d1'] = np.where(dfA['fed_d1']=='y', '1', '0')
dfA['fed_d1'] = pd.to_numeric(dfA['fed_d1'], errors='coerce')
dfA['mushy'] = np.where(dfA['mushy']=='y', '1', '0')
dfA['mushy'] = pd.to_numeric(dfA['mushy'], errors='coerce')
dfA['twin'] = np.where(dfA['twin']=='y', '1', '0')
dfA['twin'] = pd.to_numeric(dfA['twin'], errors='coerce')
dfA['comdel'] = np.where(dfA['comdel']=='y', '1', '0')
dfA['comdel'] = pd.to_numeric(dfA['comdel'], errors='coerce')
dfA['no_life'] = np.where(dfA['no_life']=='y', '1', '0')
dfA['no_life'] = pd.to_numeric(dfA['no_life'], errors='coerce')
dfA['cried'] = np.where(dfA['cried']=='y', '1', '0')
dfA['cried'] = pd.to_numeric(dfA['cried'], errors='coerce')
dfA['cyanosis'] = np.where(dfA['cyanosis']=='y', '1', '0')
dfA['cyanosis'] = pd.to_numeric(dfA['cyanosis'], errors='coerce')
dfA['move_lb'] = np.where(dfA['move_lb']=='y', '1', '0')
dfA['move_lb'] = pd.to_numeric(dfA['move_lb'], errors='coerce')
dfA['wt_loss'] = np.where(dfA['wt_loss']=='y', '1', '0')
dfA['wt_loss'] = pd.to_numeric(dfA['wt_loss'], errors='coerce')
dfA['wasting'] = np.where(dfA['wasting']=='y', '1', '0')
dfA['wasting'] = pd.to_numeric(dfA['wasting'], errors='coerce')
dfA['or_cand'] = np.where(dfA['or_cand']=='y', '1', '0')
dfA['or_cand'] = pd.to_numeric(dfA['or_cand'], errors='coerce')
dfA['rigidity'] = np.where(dfA['rigidity']=='y', '1', '0')
dfA['rigidity'] = pd.to_numeric(dfA['rigidity'], errors='coerce')
dfA['anaemia'] = np.where(dfA['anaemia']=='y', '1', '0')
dfA['anaemia'] = pd.to_numeric(dfA['anaemia'], errors='coerce')
dfA['swe_legs'] = np.where(dfA['swe_legs']=='y', '1', '0')
dfA['swe_legs'] = pd.to_numeric(dfA['swe_legs'], errors='coerce')
dfA['swell'] = np.where(dfA['swell']=='y', '1', '0')
dfA['swell'] = pd.to_numeric(dfA['swell'], errors='coerce')
dfA['swe_oral'] = np.where(dfA['swe_oral']=='y', '1', '0')
dfA['swe_oral'] = pd.to_numeric(dfA['swe_oral'], errors='coerce')
dfA['swe_oth'] = np.where(dfA['swe_oth']=='y', '1', '0')
dfA['swe_oth'] = pd.to_numeric(dfA['swe_oth'], errors='coerce')
dfA['swe_gen'] = np.where(dfA['swe_gen']=='y', '1', '0')
dfA['swe_gen'] = pd.to_numeric(dfA['swe_gen'], errors='coerce')
dfA['swe_breast'] = np.where(dfA['swe_breast']=='y', '1', '0')
dfA['swe_breast'] = pd.to_numeric(dfA['swe_breast'], errors='coerce')
dfA['swe_armp'] = np.where(dfA['swe_armp']=='y', '1', '0')
dfA['swe_armp'] = pd.to_numeric(dfA['swe_armp'], errors='coerce')
dfA['swe_neck'] = np.where(dfA['swe_neck']=='y', '1', '0')
dfA['swe_neck'] = pd.to_numeric(dfA['swe_neck'], errors='coerce')
dfA['lab_24'] = np.where(dfA['lab_24']=='y', '1', '0')
dfA['lab_24'] = pd.to_numeric(dfA['lab_24'], errors='coerce')
dfA['died_lab'] = np.where(dfA['died_lab']=='y', '1', '0')
dfA['died_lab'] = pd.to_numeric(dfA['died_lab'], errors='coerce')
dfA['death_24'] = np.where(dfA['death_24']=='y', '1', '0')
dfA['death_24'] = pd.to_numeric(dfA['death_24'], errors='coerce')
dfA['baby_al'] = np.where(dfA['baby_al']=='y', '1', '0')
dfA['baby_al'] = pd.to_numeric(dfA['baby_al'], errors='coerce')
dfA['baby_pos'] = np.where(dfA['baby_pos']=='y', '1', '0')
dfA['baby_pos'] = pd.to_numeric(dfA['baby_pos'], errors='coerce')
dfA['mon_early'] = np.where(dfA['mon_early']=='y', '1', '0')
dfA['mon_early'] = pd.to_numeric(dfA['mon_early'], errors='coerce')
dfA['breast_fd'] = np.where(dfA['breast_fd']=='y', '1', '0')
dfA['breast_fd'] = pd.to_numeric(dfA['breast_fd'], errors='coerce')
dfA['del_cs'] = np.where(dfA['del_cs']=='y', '1', '0')
dfA['del_cs'] = pd.to_numeric(dfA['del_cs'], errors='coerce')
dfA['prof_ass'] = np.where(dfA['prof_ass']=='y', '1', '0')
dfA['prof_ass'] = pd.to_numeric(dfA['prof_ass'], errors='coerce')
dfA['del_ass'] = np.where(dfA['del_ass']=='y', '1', '0')
dfA['del_ass'] = pd.to_numeric(dfA['del_ass'], errors='coerce')
dfA['del_norm'] = np.where(dfA['del_norm']=='y', '1', '0')
dfA['del_norm'] = pd.to_numeric(dfA['del_norm'], errors='coerce')
dfA['del_fac'] = np.where(dfA['del_fac']=='y', '1', '0')
dfA['del_fac'] = pd.to_numeric(dfA['del_fac'], errors='coerce')
dfA['del_home'] = np.where(dfA['del_home']=='y', '1', '0')
dfA['del_home'] = pd.to_numeric(dfA['del_home'], errors='coerce')
dfA['del_else'] = np.where(dfA['del_else']=='y', '1', '0')
dfA['del_else'] = pd.to_numeric(dfA['del_else'], errors='coerce')
dfA['vb_bet'] = np.where(dfA['vb_bet']=='y', '1', '0')
dfA['vb_bet'] = pd.to_numeric(dfA['vb_bet'], errors='coerce')
dfA['vb_men'] = np.where(dfA['vb_men']=='y', '1', '0')
dfA['vb_men'] = pd.to_numeric(dfA['vb_men'], errors='coerce')
dfA['vb_after'] = np.where(dfA['vb_after']=='y', '1', '0')
dfA['vb_after'] = pd.to_numeric(dfA['vb_after'], errors='coerce')
dfA['hair'] = np.where(dfA['hair']=='y', '1', '0')
dfA['hair'] = pd.to_numeric(dfA['hair'], errors='coerce')
dfA['exc_drink'] = np.where(dfA['exc_drink']=='y', '1', '0')
dfA['exc_drink'] = pd.to_numeric(dfA['exc_drink'], errors='coerce')
dfA['paral_one'] = np.where(dfA['paral_one']=='y', '1', '0')
dfA['paral_one'] = pd.to_numeric(dfA['paral_one'], errors='coerce')
dfA['eye_sunk'] = np.where(dfA['eye_sunk']=='y', '1', '0')
dfA['eye_sunk'] = pd.to_numeric(dfA['eye_sunk'], errors='coerce')
dfA['bl_orif'] = np.where(dfA['bl_orif']=='y', '1', '0')
dfA['bl_orif'] = pd.to_numeric(dfA['bl_orif'], errors='coerce')
dfA['diff_sw'] = np.where(dfA['diff_sw']=='y', '1', '0')
dfA['diff_sw'] = pd.to_numeric(dfA['diff_sw'], errors='coerce')
dfA['not_preg'] = np.where(dfA['not_preg']=='y', '1', '0')
dfA['not_preg'] = pd.to_numeric(dfA['not_preg'], errors='coerce')
dfA['pregnant'] = np.where(dfA['pregnant']=='y', '1', '0')
dfA['pregnant'] = pd.to_numeric(dfA['pregnant'], errors='coerce')
dfA['more4'] = np.where(dfA['more4']=='y', '1', '0')
dfA['more4'] = pd.to_numeric(dfA['more4'], errors='coerce')
dfA['pend_6w'] = np.where(dfA['pend_6w']=='y', '1', '0')
dfA['pend_6w'] = pd.to_numeric(dfA['pend_6w'], errors='coerce')
dfA['multip'] = np.where(dfA['multip']=='y', '1', '0')
dfA['multip'] = pd.to_numeric(dfA['multip'], errors='coerce')
dfA['cs_prev'] = np.where(dfA['cs_prev']=='y', '1', '0')
dfA['cs_prev'] = pd.to_numeric(dfA['cs_prev'], errors='coerce')
dfA['del_6wks'] = np.where(dfA['del_6wks']=='y', '1', '0')
dfA['del_6wks'] = pd.to_numeric(dfA['del_6wks'], errors='coerce')
dfA['first_p'] = np.where(dfA['first_p']=='y', '1', '0')
dfA['first_p'] = pd.to_numeric(dfA['first_p'], errors='coerce')

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

print('recod the cause of death categories...')
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

print(dfA_subA['broad_cause_cat'].value_counts())

#keep the two categories
#dfA_subB = dfA_subA[(dfA_subA['broad_cause_cat'] == 'HIV/AIDS & TB') | 
#                    (dfA_subA['broad_cause_cat'] == 'Other infectious')]
dfA_subB = dfA_subA[(dfA_subA['broad_cause_cat'] != '')]

print('encoding the labels in the dataset, please wait...')
#Turning labels into numbers
LE = LabelEncoder()
dfA_subB['label_num'] = LE.fit_transform(dfA_subB['broad_cause_cat'])
print(dfA_subB.head())

# saving the dataframe
dfA_subB.to_csv('StatsTests.csv')

# saving the dataframe
dfA_subB.to_csv(r'C:\Users\a0056407\Desktop\StatsTests\file3.csv', index=False)

print('generating features, please wait...')
#splitting the data into features and targets
X1 = dfA_subB[dfA_subB.columns[~dfA_subB.columns.isin(['icdmain_consensus','broad_cause_cat', 
                                                      'label_num','id', 'disease_description'])]]
print (X1.shape)
#tfidf vectorizer with 1 and 2 grams
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df = 2, max_df = .95)
print('generating features using tfidf vectorizor, please wait...')
#splitting the data into features and targets
X2 = tfidf_vectorizer.fit_transform(dfA_subB['disease_description'].values.astype(str)) #features
print (X2.shape)

X = hstack([X1, X2])

y = dfA_subB['label_num'].values #target
print (X.shape)
print(y.shape)

# from imblearn.over_sampling import RandomOverSampler
# ros = RandomOverSampler(random_state=0)
# X_resampled, y_resampled = ros.fit_resample(X, y)

print('balancing the data')
# # transform the dataset
strategy = {0:3388, 1:3388, 2:3388, 3:3388, 4:3388, 5:3388, 6:3388, 7:3388, 8:3388, 9:3388, 10:3388, 11:3388}
ros = RandomOverSampler(sampling_strategy=strategy)
X, y = ros.fit_resample(X, y)

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
                                              min_samples_leaf=1,
                                              min_samples_split=2, min_weight_fraction_leaf=0.0,
                                              random_state=None, splitter='best')

xgboost_best_model =  XGBClassifier(objective="multi:softproba", alpha=0, n_estimators=20, random_state=42, num_class = 11 ,                                   learning_rate=0.1,
                                  max_depth=10, eval_metric='mlogloss')

nbc_best_model = GaussianNB(var_smoothing=2e-9)

rf_best_model =RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                              class_weight=None,
                                              criterion='gini', max_depth=None,
                                              max_features='auto',
                                              max_leaf_nodes=None,
                                              max_samples=None,
                                              min_impurity_decrease=0.0,
                                             
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
y_score = lr_classifier.fit(X_train_b, y_train_b).predict(X_test_b)
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
y_score = ann_classifier.fit(X_train_b, y_train_b).predict(X_test_b)
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
y_score = dt_classifier.fit(X_train_b, y_train_b).predict(X_test_b)
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
y_score = xgb_classifier.fit(X_train_b, y_train_b).predict(X_test_b)
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
y_score = nbc_classifier.fit(X_train_b, y_train_b).predict(X_test_b)
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
y_score = knn_classifier.fit(X_train_b, y_train_b).predict(X_test_b)
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
y_score = svm_classifier.fit(X_train_b, y_train_b).predict(X_test_b)
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
y_score = rf_classifier.fit(X_train_b, y_train_b).predict(X_test_b)
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

