# IntelligentDataMapper-and-DataModeller

Intelligent Data Mapping and Data Model
Summary
How ML Algorithms can boost the turnaround time for data mapping and data modelling for Data Architects? We will build a semantic matching utility that uses the glossary (column name & description) provided for each of the source column and predicts the right target tables/columns run by proven similarity measures algorithms. 
Machine Learning Algorithms
1.	FUZZY WUZZY: 
Fuzzy string matching uses Levenshtein Distance to calculate the differences between sequences in a simple-to-use package. The Levenshtein distance is a metric to measure how apart are two sequences of words. In other words, it measures the minimum number of edits that you need to do to change a one-word sequence into the other. These edits can be insertions, deletions or substitutions.  The maximum matching between two sentences is a score of 100. 
2.	NORMALISED WORD MEASURE DISTANCE:
WMD is a method that allows us to assess the "distance" between two documents in a meaningful way, even when they have no words in common. It uses word2vec vector embedding’s of words. It been shown to outperform many of the state-of-the-art methods in k-nearest neighbours classification.
WMD is illustrated below for two very similar sentences. The sentences have no words in common, but by matching the relevant words, WMD is able to accurately measure the (dis)similarity between the two sentences. The method also uses the bag-of-words representation of the documents (simply put, the word's frequencies in the documents), noted as dd in the figure below. The intuition behind the method is that we find the minimum "traveling distance" between documents, in other words the most efficient way to "move" the distribution of document 1 to the distribution of document 2.

In our methods, we have used Google Pre-trained Gensim model (GoogleNews-vectors-negative300.bin.gz) which has already trained and vectorised with billions of words available in google. This model can be imported and used for our incoming sentence vs sentence matching. 
NORMALISING THE VECTOR:
	model.init_sims(replace=True). This command makes the vectors values in the model normalised so that its length does not affect the distance measures and makes it more accurate.

 
Above figure explains how the word embedding used to find the semantic relationships using their distance in vector space.
3.	COSINE LAW:
Cosine similarity is a common vector based matching method whereby the input string is transformed into vector space so that the Euclidean cosine rule can be used to determine similarity.
In this measure, we will get the cosine angle between the two vectors that is being matched, the more it has semantically close the lesser the angle between them. Using this logic, we can get the words closer to each other semantically.
Features of the solution:
Aforementioned algorithms will be deployed to find the best match for the incoming data sources vs the Existing Data model for data mapping purposes.
1.	MATCHING COLUMNS AND TABLE NAMES IN SOURCE & TARGET:
The new incoming data sources should have the below attributes which can be matched with the target attributes.
Source Column Name|Source Table Name | Business Glossary Source
The Target dataset with already mapped information consist of information like below:
Target Column Name | Target Table Name | Business Glossary Target
Now, the business glossary from the source sheet is validated against with the target glossary using the above-mentioned algorithms. Based on the matching score (Highest Fuzzy Wuzzy,Lowest for Norm WMD and Cosine) between them we can fix the most matched target column and table name for each source column name.
We have benchmarked 85% percent accuracy for our modelling.
2.	MATCHING NEW TABLES/COLUMNS FROM SOURCE WHICH ARE NOT AVAILABLE IN TARGET:
a.	New Columns/Tables:
For any new columns, which is not part of the existing data model, we can have a corpus file with all necessary standards and propose a new column name and table name, which can be fit in, to our existing data model. For this feature, we can have confidence interval for matching score values, separate the columns with very less score values, consider them as new, and propose the new column and tables names based on the corpus file.
b.	Appending New Tables to existing DM:
Once we have the list of new tables and columns, we can merge the tables into Existing DM by using the foreign keys present in them. 
For this, we first need to load the existing DM information into Python Graphs, which denotes the linking between them using edges and vertices. 
Next, we can add new vertices for the new tables and add it to our existing graph representing the data Model.
For Ex:
The below fig, show the graphical representation in python for the sample customer DM. 
 
If an new entity is planned to added we can refer the foreign key present in them against with the primary key present in existing DM, and do the matching and based on the matching score it will be associated with the best matched table to be joined with. It is also possible to be matched with many other tables as well by having a confidence interval around the matching.
For instance let us say, Customer Service table is coming up with matching for contact and Dealer. Now the graph after matching will look like,

 
By this way, we can also add new tables into existing DM as well. It will considerably reduce the effort spent in Data mapping and Data Modelling as well.
3.	SCORING
All the predictions of table/column will be provided with propensity scores for their match between source and target. Also upto three column names as matches will be provided enabling the database developers to pick the most suitable ones in a jiffy. 
