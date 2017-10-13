

# Tronc Data Modeling Take Home Test:
# Predicting the Performance of Facebook Posts

The jupyter noteboook walks through a first iteration of attempting to predict the 'success' of facebook posts. Additionally there is a python app which
will do the ETL and model validation and print results to standard out.We proxy success by using 'post_impressions_organic_unique' insights, and develop a feature space by processing the textual components of the post using TF_IDF Vectorization and Latent Dirichlet Allocation (LDA) into a vector expressing the probabilistic relevance to a set of known 'topics.' The intuition is that certain topics will draw more interest than others and hence more shares and subsequent impressions.

We will fit a Random Forest model to the transformed dataset and collect standard regression metrics such as Explained Variance and Mean Squared Error.

### Libraries Used:

 - numpy
 - pandas
 - matplotlib
 - nltk
 - scipy
 - sklearn
 - nltk (requires corpus download - 'nltk.download()')
 
 also requires the file 'posts.json' be in the working directory.
 

 
 


