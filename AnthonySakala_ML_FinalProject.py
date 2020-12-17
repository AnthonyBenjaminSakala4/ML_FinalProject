#!/usr/bin/env python
# coding: utf-8

# In this notebook, I will attempt at implementing a few recommendation algorithms (content based, popularity based 
# and collaborative filtering) and try to build an ensemble of these models to come up with our final recommendation system. With us, we have two MovieLens datasets.
# 
# The Full Dataset: Consists of 26,000,000 ratings and 750,000 tag applications applied to 45,000 movies by 270,000 users. Includes tag genome data with 12 million relevance scores across 1,100 tags.
# 
# The Small Dataset: Comprises of 100,000 ratings and 1,300 tag applications applied to 9,000 movies by 700 users.
# 
# I will build a Simple Recommender using movies from the Full Dataset whereas all personalised recommender systems will make use of the small dataset (due to the computing power I possess being very limited). As a first step, I will build my simple recommender system.
# 

# In[1]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD 
# use cross_validate for evaluate
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import KFold
from surprise import NormalPredictor

import warnings; warnings.simplefilter('ignore')
import streamlit as st


# In[2]:


dataset = pd.read_csv('datasets/movies_metadata.csv')
#dataset.head()


# In[3]:


dataset['genres'] = dataset['genres'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# I use the TMDB Ratings to come up with our Top Movies Chart. I will use IMDB's weighted rating formula to construct my chart. Mathematically, it is represented as follows:
# 
# Weighted Rating (WR) =  (v/v+m.R)+(m/v+m.C) 
# where,
# 
# v is the number of votes for the movie
# m is the minimum votes required to be listed in the chart
# R is the average rating of the movie
# C is the mean vote across the whole report
# The next step is to determine an appropriate value for m. I will use 95th percentile as our cutoff. In other words, for a movie to feature in the charts, it must have more votes than at least 95% of the movies in the list.
# 
# I will build our overall Top 250 Chart and will define a function to build charts for a particular genre. Let's begin!

# In[4]:


vote_counts = dataset[dataset['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = dataset[dataset['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
#C


# In[5]:


m = vote_counts.quantile(0.95)
#m


# In[6]:


dataset['year'] = pd.to_datetime(dataset['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[7]:


qualified = dataset[(dataset['vote_count'] >= m) & (dataset['vote_count'].notnull()) & (
    dataset['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
#qualified.shape


# Therefore, to qualify to be considered for the chart, a movie has to have at least 434 votes on TMDB. 
# We also see that the average rating for a movie on TMDB is 5.244 on a scale of 10. 2274 Movies qualify 
# to be on our chart.

# In[8]:


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[9]:


qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)


# Top Movies

# In[10]:


#qualified.head(15)


# We see that the top three movies on the chart, Inception, The Dark Knight and Interstellar are from 
# Christopher Nolan Films. Moveover, The chart also indicates a strong bias of TMDB Users towards particular 
# genres and directors.
# 
# Let us now construct a function that builds charts for particular genres. For this, I will use or relax our 
# default conditions to the 85th percentile instead of 95.

# In[11]:


s = dataset.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_dataset = dataset.drop('genres', axis=1).join(s)


# In[12]:


def build_chart(genre, percentile=0.85):
    df = gen_dataset[gen_dataset['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (
        df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (
        m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified


# Let us see our method in action by displaying the Top 15 Romance Movies 
# (Romance almost didn't feature at all in our Generic Top Chart despite being one of the most popular movie genres).

# Top Romance Movies

# In[13]:


#build_chart('Romance').head(15)


# The top romance movie according to our metrics is Dilwale Dulhania Le Jayenge.

# Content Based Recommender
# 
# The chart we built earlier has some severe limitations. For one, it didn't consider on romantic movies. If a person who loves romantic movies (and hates action) were to look at our Top 15 Chart, s/he wouldn't probably like most of the movies. If s/he were to go one step further and look at our charts by genre, s/he wouldn't still be getting the best recommendations.
# 
# For instance, consider a person who loves Dilwale Dulhania Le Jayenge.
# One inference we can obtain is that what if some people loves such movies. Even if 
# s/he were to access the romance chart, s/he wouldn't find these as the top recommendations.
# 
# To personalise our recommendations more, I am going to build an engine that computes similarity between movies based 
# on certain metrics and suggests movies that are most similar to a particular movie that a user liked. Since we will 
# be using movie metadata (or content) to build this engine, this also known as Content Based Filtering.
# 
# I will build two Content Based Recommenders based on:
# 
# Movie Overviews and Taglines
# Movie Cast, Crew, Keywords and Genre
# Also, as mentioned in the introduction, I will be using a subset of all the movies available to us due to limiting 
# computing power available to me.

# In[14]:


links_small = pd.read_csv('datasets/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')


# Before we are able to perform any mergers of the main dataframe, we need to make sure that the ID column of our 
# main dataframe is clean and of type integer. To do this, let us try to perform an integer conversion of our IDs 
# and if an exception is raised, we will replace the ID with NaN. We will then proceed to drop these rows from our 
# dataframe.

# In[15]:


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[16]:


dataset['id'] = dataset['id'].apply(convert_int)


# In[17]:


# dataset[dataset['id'].isnull()]


# In[18]:



links_small_data = dataset.drop([19730, 29503, 35587])


# In[19]:


links_small_data['id'] = links_small_data['id'].astype('int')


# In[20]:


s_links_small_data = links_small_data[links_small_data['id'].isin(links_small)]
#s_links_small_data.shape


# We have 9099 movies avaiable in our small movies metadata dataset which is 5 times smaller than our original 
# dataset of 45000 movies.

# Movie Description Based Recommender
# 
# Let us first try to build a recommender using movie descriptions and taglines. We do not have a quantitative metric 
# to judge our machine's performance so this will have to be done qualitatively.

# In[21]:


s_links_small_data['tagline'] = s_links_small_data['tagline'].fillna('')
s_links_small_data['description'] = s_links_small_data['overview'] + s_links_small_data['tagline']
s_links_small_data['description'] = s_links_small_data['description'].fillna('')


# In[22]:


tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(s_links_small_data['description'])

#tfidf_matrix.shape


# Cosine Similarity
# 
# I will be using the Cosine Similarity to calculate a numeric quantity that denotes the similarity between two 
# movies. Mathematically, it is defined as follows:
# 
# cosine(x,y)=x.y⊺/||x||.||y|| 
# 
# Since we have used the TF-IDF Vectorizer, calculating the Dot Product will directly give us the Cosine Similarity 
# Score. Therefore, we will use sklearn's linear_kernel instead of cosine_similarities since it is much faster.

# In[23]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#cosine_sim[0]


# We now have a pairwise cosine similarity matrix for all the movies in our dataset. The next step is to write a 
# function that returns the 30 most similar movies based on the cosine similarity score.

# In[24]:


s_links_small_data = s_links_small_data.reset_index()
titles = s_links_small_data['title']
indices = pd.Series(s_links_small_data.index, index=s_links_small_data['title'])


# In[25]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# Let us now try and get the top recommendations for a few movies and see how good the recommendations are.

# In[26]:


#get_recommendations('The Dark Knight').head(10)


# In[27]:


#get_recommendations('The Godfather').head(10)


# We see that for The Dark Knight, our system is able to identify it as a Batman film and subsequently recommend 
# other Batman films as its top recommendations. But unfortunately, that is all this system can do at the moment. 
# This is not of much use to most people as it doesn't take into considerations very important features such as cast, 
# crew, director and genre, which determine the rating and the popularity of a movie. Someone who liked 
# The Dark Knight probably likes it more because of Nolan and would hate Batman Forever and every other substandard 
# movie in the Batman Franchise.
# 
# Therefore, we are going to use much more suggestive metadata than Overview and Tagline. In the next subsection, 
# we will build a more sophisticated recommender that takes genre, keywords, cast and crew into consideration.

# Metadata Based Recommender
# 
# To build our standard metadata based content recommender, we will need to merge our current dataset with the crew 
# and the keyword datasets. Let us prepare this data as our first step.

# In[28]:


credits = pd.read_csv('datasets/credits.csv')
keywords = pd.read_csv('datasets/keywords.csv')


# In[29]:


keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
links_small_data['id'] = links_small_data['id'].astype('int')


# In[30]:


#links_small_data.shape


# In[31]:


links_small_data = links_small_data.merge(credits, on='id')
links_small_data = links_small_data.merge(keywords, on='id')


# In[32]:


s_links_small_data = links_small_data[links_small_data['id'].isin(links_small)]

#s_links_small_data.shape


# We now have our cast, crew, genres and credits, all in one dataframe. Let us wrangle this a little more using the 
# following intuitions:
# 
# Crew: From the crew, we will only pick the director as our feature since the others don't contribute that much to 
#     the feel of the movie.
#     
# Cast: Choosing Cast is a little more tricky. Lesser known actors and minor roles do not really affect people's 
#     opinion of a movie. Therefore, we must only select the major characters and their respective actors. 
#     Arbitrarily we will choose the top 3 actors that appear in the credits list.

# In[33]:


s_links_small_data['cast'] = s_links_small_data['cast'].apply(literal_eval)
s_links_small_data['crew'] = s_links_small_data['crew'].apply(literal_eval)
s_links_small_data['keywords'] = s_links_small_data['keywords'].apply(literal_eval)
s_links_small_data['cast_size'] = s_links_small_data['cast'].apply(lambda x: len(x))
s_links_small_data['crew_size'] = s_links_small_data['crew'].apply(lambda x: len(x))


# In[34]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[35]:


s_links_small_data['director'] = s_links_small_data['crew'].apply(get_director)


# In[36]:


s_links_small_data['cast'] = s_links_small_data['cast'].apply(lambda x: [i['name'] for i in x] 
                                                              if isinstance(x, list) else [])
s_links_small_data['cast'] = s_links_small_data['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)


# In[37]:


s_links_small_data['keywords'] = s_links_small_data['keywords'].apply(lambda x: [i['name'] for i in x] 
                                                                      if isinstance(x, list) else [])


# My approach to building the recommender is going to be extremely hacky. What I plan on doing is creating a metadata
# dump for every movie which consists of genres, director, main actors and keywords. I then use a Count Vectorizer to 
# create our count matrix as we did in the Description Recommender. The remaining steps are similar to what we did 
# earlier: we calculate the cosine similarities and return movies that are most similar.
# 
# These are steps I will follow in the preparation of my genres and credits data:
# 
# Strip Spaces and Convert to Lowercase from all our features. This way, our engine will not confuse people
# such as Johnny Depp and Johnny Galecki.
# 
# Mention Director 3 times to give it more weight relative to the entire cast.

# In[38]:


s_links_small_data['cast'] = s_links_small_data['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) 
                                                                         for i in x])


# In[39]:


s_links_small_data['director'] = s_links_small_data['director'].astype('str').apply(
    lambda x: str.lower(x.replace(" ", "")))
s_links_small_data['director'] = s_links_small_data['director'].apply(lambda x: [x,x, x])


# Keywords
# 
# I will do a small amount of pre-processing of our keywords before putting them to any use. As a first step, we 
# calculate the frequenct counts of every keyword that appears in the dataset.

# In[40]:


s = s_links_small_data.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'


# In[41]:


s = s.value_counts()
#s[:5]


# Keywords occur in frequencies ranging from 1 to 610. We do not have any use for keywords that occur only once. 
# Therefore, these can be safely removed. Finally, we will convert every word to its stem so that words such as Dogs 
# and Dog are considered the same.
# 

# In[42]:


s = s[s > 1]

stemmer = SnowballStemmer('english')
stemmer.stem('dogs')


# In[43]:


def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words


# In[44]:


s_links_small_data['keywords'] = s_links_small_data['keywords'].apply(filter_keywords)
s_links_small_data['keywords'] = s_links_small_data['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
s_links_small_data['keywords'] = s_links_small_data['keywords'].apply(
    lambda x: [str.lower(i.replace(" ", "")) for i in x])

s_links_small_data['soup'] = s_links_small_data['keywords'] + s_links_small_data['cast'] + s_links_small_data[
    'director'] + s_links_small_data['genres']
s_links_small_data['soup'] = s_links_small_data['soup'].apply(lambda x: ' '.join(x))


# In[45]:


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(s_links_small_data['soup'])


# In[46]:


cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[47]:


s_links_small_data = s_links_small_data.reset_index()
titles = s_links_small_data['title']
indices = pd.Series(s_links_small_data.index, index=s_links_small_data['title'])


# We will reuse the get_recommendations function that we had written earlier. Since our cosine similarity scores 
# have changed, we expect it to give us different (and probably better) results. Let us check for The Dark Knight 
# again and see what recommendations I get this time around.

# In[48]:


#get_recommendations('The Dark Knight').head(10)


# I am much more satisfied with the results I get this time around. The recommendations seem to have recognized other 
# Christopher Nolan movies (due to the high weightage given to director) and put them as top recommendations. 
# 
# We can of course experiment on this engine by trying out different weights for our features 
# (directors, actors, genres), limiting the number of keywords that can be used in the soup, weighing genres based 
# on their frequency, only showing movies with the same languages, etc.

# Let me also get recommendations for another movie, Friday the 13th

# In[49]:


#get_recommendations('Friday the 13th').head(10)


# Popularity and Ratings
# 
# One thing that we can notice about our recommendation system is that it recommends movies regardless of ratings and 
# popularity.
# 
# Therefore, we will add a mechanism to remove bad movies and return movies which are popular and have had a good 
# critical response.
# 
# I will take the top 25 movies based on similarity scores and calculate the vote of the 60th percentile movie. Then, 
# using this as the value of  m , we will calculate the weighted rating of each movie using IMDB's formula like we 
# did in the Simple Recommender section.
# 

# In[50]:


def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = s_links_small_data.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (
        movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified


# In[51]:


#improved_recommendations('The Dark Knight')


# Let me also get the recommendations for Friday the 13th, my favorite movie.

# In[52]:


#improved_recommendations('Friday the 13th')


# Collaborative Filtering
# 
# Our content based engine suffers from some severe limitations. It is only capable of suggesting movies which are close to a certain movie. That is, it is not capable of capturing tastes and providing recommendations across genres.
# 
# Also, the engine that we built is not really personal in that it doesn't capture the personal tastes and biases of a user. Anyone querying our engine for recommendations based on a movie will receive the same recommendations for that movie, regardless of who s/he is.
# 
# Therefore, in this section, we will use a technique called Collaborative Filtering to make recommendations to Movie Watchers. Collaborative Filtering is based on the idea that users similar to me can be used to predict how much I will like a particular product or service those users have used/experienced but I have not.
# 
# I will not be implementing Collaborative Filtering from scratch. Instead, I will use the Surprise library that used extremely powerful algorithms like Singular Value Decomposition (SVD) to minimise RMSE (Root Mean Square Error) and give great recommendations.
# 

# In[53]:


reader = Reader()


# In[54]:


ratings = pd.read_csv('datasets/ratings_small.csv')
# ratings.head()


# Model Evaluation and Optimization

# In[55]:



data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
cross_validate(NormalPredictor(), data, cv=5)


# In[56]:


# svd = SVD()
# cross_validate(svd, data, measures=['RMSE', 'MAE'])

# define a cross-validation iterator
kf = KFold(n_splits=5)

svd = SVD()

for trainset, testset in kf.split(data):

    # train and test algorithm.
    svd.fit(trainset)
    predictions = svd.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)


# We get a mean Root Mean Sqaure Error of 0.8926 which is more than good enough for our case. Let us now train on our dataset and arrive at predictions.

# In[57]:


trainset = data.build_full_trainset()
svd.fit(trainset)


# In[58]:


#ratings[ratings['userId'] == 1]


# In[59]:


svd.predict(1, 302, 3)


# Advanced Recommender
# 
# In this section, I will try to build a simple Advanced recommender that brings together techniques implemented in the content based and collaborative filter based engines. This is how it will work:
# 
# Input: User ID and the Title of a Movie
# Output: Similar movies sorted on the basis of expected ratings by that particular user.

# In[60]:


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[61]:


data_id_mapping = pd.read_csv('datasets/links_small.csv')[['movieId', 'tmdbId']]
data_id_mapping['tmdbId'] = data_id_mapping['tmdbId'].apply(convert_int)
data_id_mapping.columns = ['movieId', 'id']
data_id_mapping = data_id_mapping.merge(s_links_small_data[['title', 'id']], on='id').set_index('title')


# In[62]:


data_indices_mapping = data_id_mapping.set_index('id')


# In[63]:


def advanced_recommender(userId, title):
    idx = indices[title]
    tmdbId = data_id_mapping.loc[title]['id']
    #print(idx)
    movie_id = data_id_mapping.loc[title]['movieId']
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = s_links_small_data.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, data_indices_mapping.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)


# In[64]:


#advanced_recommender(5, 'Friday the 13th')


# In[65]:


#advanced_recommender(300, 'Friday the 13th')


# In[66]:


#advanced_recommender(201, 'Thor')


# In[67]:
import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('image/background1.png')


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

st.title('Machine Learning Movie Recommender System')
id_text = st.text('Enter user ID:')
user_id = st.number_input("", 1, 200000, step=1)

movie_text = st.text('Enter the name of the movie:')
movie_name = st.text_input("")
button_recommend = st.button("GO")

if button_recommend:
    list_of_movies = advanced_recommender(user_id, movie_name)
    st.write(list_of_movies)



# In[ ]:





# In[ ]:




