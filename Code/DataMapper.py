#####################
# DataMappingCode.py
# ------------------
#####################
# Attribution Information: Data Mapping python code enables users to pre-map the incoming 
# new data sources into their existing data model. It requires info about the existing data model 
# with business glossary as a data file and incoming data will be mapped based on it.
# We get the mapped target tables and columns for Four distance measures(Fuzzy, Cosine, Jaccard, Norm Wmd) 
# which can be used by ETL developers as base of their data mappings.
# Thereby reducing the manual work behind it. 
# Predefined Column Required in Source Data file:
# 1. Entity Table Name : this represents the Entity or table name behind the existing column. Ideally represents what sort of data is stored
# 2. Attribute Name : Short or long description about the column 
# 3. Column Name :  Target Column Name
# 4. Table Name : Target Table Name 
# 5. ID :  Unique identifier for each row
# Predefined column required in New data source file:
# 1. Entity Table Name : Represents the incoming column data entity. For ex: Either its a customer or Dealer related column
# 2. Attribute Name : Short or long description about the column
# 3. Source Column : new Source column Name
# 4. Source Table/File Name : New Source Table or File name.
# Make sure the Column names are mentioned as given

# Created by Seshadri Senthamaraikannan - Cognizant Technology Solutions Belgium.


import argparse
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine,jaccard
from nltk import word_tokenize
stop_words = stopwords.words('english')


# The below function represents the Normalised Word Measure Distance where in two words/Sentences distance difference is calculated in vector space	using a predefined google model

def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)

# The below function represents the Senctence to vector convertion, it is used to convert the text to vector for distance measurement

def sentence2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

# The below set of functions used to calculate score using the distance algorithms. Using entity table name and Attribute Name (Desc about the column) as a matching criteria we will generate the difference in distance between them in vector space and most closest values with its corresponding score value is returned out of the function.


def NormWmdScore(dfcomp,dfFull):
    ### Normalised WMD Score Calculation is implemented in this function.	
    Mynormwmdlist = []
    MyFinalList=[]
    for index,newrows in dfcomp.iterrows():
        innerlist = []
        Entity = newrows['Entity Table Name']
        if Entity in dfFull['Entity Table Name']:
            Entity_data = dfFull.loc[dfFull['Entity Table Name']==Entity]
            for index,srcrows in Entity_data.iterrows():
                Score = norm_wmd(newrows['Attribute Name'],srcrows['Attribute Name'])
                innerlist.append([newrows['ID'],Score,srcrows['Column Name'],srcrows['Table Name']])
        else:
            for index,srcrows in dfFull.iterrows():
                Score = norm_wmd(newrows['Attribute Name'],srcrows['Attribute Name'])
                innerlist.append([newrows['ID'],Score,srcrows['Column Name'],srcrows['Table Name']])
        Mynormwmdlist.append(innerlist)
    for i in range(len(Mynormwmdlist)):
        FirstVal=min(Mynormwmdlist[i],key=lambda x:x[1])
        Mynormwmdlist[i].remove(FirstVal)
        SecondVal=min(Mynormwmdlist[i],key=lambda x:x[1])
        Mynormwmdlist[i].remove(SecondVal)
        ThirdVal=min(Mynormwmdlist[i],key=lambda x:x[1])
        MyFinalList.append([FirstVal[0],FirstVal[1],FirstVal[2],FirstVal[3],SecondVal[2],SecondVal[3],ThirdVal[2],ThirdVal[3]])
       
    return MyFinalList

def CosineScore(dfcomp,dfFull):		
    ### Cosine Score calculation is implemented in this function.
    Mycosinelist = []
    MyFinalList=[]
    for index,newrows in dfcomp.iterrows():
        innerlist = []
        Entity = newrows['Entity Table Name']
        if Entity in dfFull['Entity Table Name']:
            Entity_data = dfFull.loc[dfFull['Entity Table Name']==Entity]            
            for index,srcrows in Entity_data.iterrows():
                Score = cosine(sentence2vec(newrows['Attribute Name']),sentence2vec(srcrows['Attribute Name']))
                innerlist.append([newrows['ID'],Score,srcrows['Column Name'],srcrows['Table Name']])
        else:
            for index,srcrows in dfFull.iterrows():
                Score = cosine(sentence2vec(newrows['Attribute Name']),sentence2vec(srcrows['Attribute Name']))
                innerlist.append([newrows['ID'],Score,srcrows['Column Name'],srcrows['Table Name']])
        Mycosinelist.append(innerlist)
    for i in range(len(Mycosinelist)):
        FirstVal=min(Mycosinelist[i],key=lambda x:x[1])
        Mycosinelist[i].remove(FirstVal)
        SecondVal=min(Mycosinelist[i],key=lambda x:x[1])
        Mycosinelist[i].remove(SecondVal)
        ThirdVal=min(Mycosinelist[i],key=lambda x:x[1])
        MyFinalList.append([FirstVal[0],FirstVal[1],FirstVal[2],FirstVal[3],SecondVal[2],SecondVal[3],ThirdVal[2],ThirdVal[3]])
        
    return MyFinalList

def JaccardScore(dfcomp,dfFull):		
    ### Jaccard distance score Value	is generated in this function
    Mycosinelist = []
    MyFinalList=[]
    for index,newrows in dfcomp.iterrows():
        innerlist = []
        Entity = newrows['Entity Table Name']
        if Entity in dfFull['Entity Table Name']:
            Entity_data = dfFull.loc[dfFull['Entity Table Name']==Entity]            
            for index,srcrows in Entity_data.iterrows():
                Score = jaccard(sentence2vec(newrows['Attribute Name']),sentence2vec(srcrows['Attribute Name']))
                innerlist.append([newrows['ID'],Score,srcrows['Column Name'],srcrows['Table Name']])
        else:
            for index,srcrows in dfFull.iterrows():
                Score = cosine(sentence2vec(newrows['Attribute Name']),sentence2vec(srcrows['Attribute Name']))
                innerlist.append([newrows['ID'],Score,srcrows['Column Name'],srcrows['Table Name']])
        Mycosinelist.append(innerlist)
    for i in range(len(Mycosinelist)):
        FirstVal=min(Mycosinelist[i],key=lambda x:x[1])
        Mycosinelist[i].remove(FirstVal)
        SecondVal=min(Mycosinelist[i],key=lambda x:x[1])
        Mycosinelist[i].remove(SecondVal)
        ThirdVal=min(Mycosinelist[i],key=lambda x:x[1])
        MyFinalList.append([FirstVal[0],FirstVal[1],FirstVal[2],FirstVal[3],SecondVal[2],SecondVal[3],ThirdVal[2],ThirdVal[3]])
        
    return MyFinalList
	


def FuzzyWuzzyScore(dfcomp,dfFull):		
    ### Fuzzy Matching score is generated out of this function.
    Mylist = []
    MyFinalList=[]
    for index,newrows in dfcomp.iterrows():
        innerlist = []
        Entity = newrows['Entity Table Name']
        if Entity in dfFull['Entity Table Name']:
            Entity_data = dfFull.loc[dfFull['Entity Table Name']==Entity]            
            for index,srcrows in Entity_data.iterrows():                
                Score = fuzz.token_set_ratio(newrows['Attribute Name'],srcrows['Attribute Name'])
                innerlist.append([newrows['ID'],Score,srcrows['Column Name'],srcrows['Table Name']])
        else:
            for index,srcrows in dfFull.iterrows():
                Score = fuzz.token_set_ratio(newrows['Attribute Name'],srcrows['Attribute Name'])
                innerlist.append([newrows['ID'],Score,srcrows['Column Name'],srcrows['Table Name']])
        Mylist.append(innerlist)

    for i in range(len(Mylist)):
        FirstVal=min(Mylist[i],key=lambda x:x[1])
        Mylist[i].remove(FirstVal)
        SecondVal=min(Mylist[i],key=lambda x:x[1])
        Mylist[i].remove(SecondVal)
        ThirdVal=min(Mylist[i],key=lambda x:x[1])
        MyFinalList.append([FirstVal[0],FirstVal[1],FirstVal[2],FirstVal[3],SecondVal[2],SecondVal[3],ThirdVal[2],ThirdVal[3]])
        
    return MyFinalList
    
def Diff(li1, li2): 
    counter=0
    for i in range(len(li1)):
        if li1[i] in li2:
            counter=counter+1
    Total=len(li2)
    Perct=(counter/Total)*100    
    return Perct
	
def main():
	#This program predicts the matching values of incoming column with Existing TGT Data Model")
	##Uploading Google New model
	
	parser = argparse.ArgumentParser(description='Automatic Data Mapping for Data Integration')
	parser.add_argument('-d','--datadir',type=str,required=True,help='Data Directory of the Data Model File')
	parser.add_argument('-m','--modeldir',type=str,required=True,help='Google Model Directory Location')
	parser.add_argument('-n','--newsrcdir',type=str,required=True,help='New Source Data location')
	parser.add_argument('-o','--ouputvalid',type=str,required=True,help='Final Output file used for Validation')
	args = parser.parse_args()
	global model
	model = gensim.models.KeyedVectors.load_word2vec_format(args.modeldir, binary=True)
	model.init_sims(replace=True)
	
	print("Imported Model")
	#Reading and loading source Data
	SrcData = pd.read_csv(args.datadir)
	SrcData['Entity Table Name']=SrcData['Entity Table Name'].str.lower()
	NewData = pd.read_csv(args.newsrcdir)
	NewData['Entity Table Name']=NewData['Entity Table Name'].str.lower()
	
	#Calcuating Score values for the test data
	FuzzyMatched = FuzzyWuzzyScore(NewData,SrcData)
	JaccardMatched = JaccardScore(NewData,SrcData)
	NormWmdMatched = NormWmdScore(NewData,SrcData)
	CosineMatched = CosineScore(NewData,SrcData)
	
	print('Matching scores calculated')
	## Converting score values to a DataFrame
	FuzzyDF = pd.DataFrame(FuzzyMatched,columns=['ID','FuzzyScore','TGTColumnNameFuzz',
								'TGTTableNameFuzz','TGTColumnName2ndFuzz','TGTTableName2ndFuzz','TGTColumnName3rdFuzz','TGTTableName3rdFuzz'])
	JaccardDF = pd.DataFrame(JaccardMatched,columns=['ID','JaccardScore','TGTColumnNameJD','TGTTableNameJD'
								,'TGTColumnName2ndJD','TGTTableName2ndJD','TGTColumnName3rdJD','TGTTableName3rdJD'])
	NormWmdDF = pd.DataFrame(NormWmdMatched,columns=['ID','NormWMDScore',
								'TGTColumnNameNorm','TGTTableNameNorm','TGTColumnName2ndNorm','TGTTableName2ndNorm','TGTColumnName3rdNorm','TGTTableName3rdNorm'])
	CosinedDF = pd.DataFrame(CosineMatched,columns=['ID','CosineScore',
								'TGTColumnNameCos','TGTTableNameCos','TGTColumnName2ndCos','TGTTableName2ndCos','TGTColumnName3rdCos','TGTTableName3rdCos'])

	Fuzz = pd.merge(NewData,FuzzyDF,how='inner',left_on='ID',right_on='ID')
	FZWMD = pd.merge(Fuzz,JaccardDF,how='inner',left_on='ID',right_on='ID')
	FZNMWMD = pd.merge(FZWMD,NormWmdDF,how='inner',left_on='ID',right_on='ID')
	FZNMWMDCS = pd.merge(FZNMWMD,CosinedDF,how='inner',left_on='ID',right_on='ID')
	FZNMWMDCS.to_csv(args.ouputvalid)
	
	if "Column Name" in NewData.columns:	
		Actual = SrcData["Column Name"].tolist()
		FuzzPred = FuzzyDF["TGTColumnNameFuzz"].tolist()
		JaccardPred=WmdDF["TGTColumnNameJD"].tolist()
		NormWmdPred=NormWmdDF["TGTColumnNameNorm"].tolist()
		CosinePred=CosinedDF["TGTColumnNameCos"].tolist()
		
		
		##Comparing Actual And Predicted Value and the below function returns the accuracy value
		FuzzAccuracy=Diff(FuzzPred,Actual)
		JaccardAccuracy=Diff(JaccardPred,Actual)
		NormWmdAcc=Diff(NormWmdPred,Actual)
		CosineAcc=Diff(CosinePred,Actual)
		
		
		## Printing out the accuracy value for the test data evaluated
		print("FuzzyLogic Accuracy : {}".format(FuzzAccuracy))
		print("Jaccard Accuracy: {}".format(JaccardAccuracy))
		print("NormWMD Accuracy: {}".format(NormWmdAcc))
		print("Cosine Accuracy: {}".format(CosineAcc))

if __name__ == "__main__":
    main()
