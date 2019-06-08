---
nav_include: 3
title: Recommender Systems
notebook: Recommender Model.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}



## Content-based Filtering
Content-based filtering collects information describing the item content and recommend items that are similar to items the user likes. The items are usually represented by n-dimensional feature vectors and the features they contain can be collected automatically, e.g., by extracting features from the audio signal, or in our case we used Spotify API to collect audio features. 

Content-based recommendation relies heavily on audio content analysis and thus content-based music recommenders exploit traditional music information techniques.

In the music domain, content-based filtering ranks songs based on how similar they are to a seed song according to some similarity measure, which focuses on an objective distance between items and does not include any subjective factors. This makes it possible to recommend new items that do not have any user ratings associated with them.

There exists a wide variety of approaches for measuring the similarity between songs.


**Some disadvantages** of Content-based filtering are:
- Cold start problem- Content-based recommender system needs time to adapt to new user's preferences. However, the cold start problem does not exist for new items.
- Gray sheep- Content-based recommendation relies heavily on what kind of items are in the collection, and the collection may be biased towards a specific user tastes. Thus, it is possible that users with atypical tastes will not receive relevant recommendations.
- Novelty- Users can receive recommendations that are too similar when the similarity function is accurate. Content-based recommender systems should promote eclecticness of items by using other factors.
- Feature limitation- Content-based recommender system are limited by the features that can be extracted from the content. In other words, the recommender system is limited by the descriptive data that is available.
- Modeling user preferences- Content similarity cannot fully capture the user&apos;s preferences, which results in a semantic gap between the user&apos;s perception of music and the music representation of the system. 



**Approach**:

Here our model will measure similarity bectween feature vectors by using cosine similarity.

**Cosine similarity** is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. The cosine of 0 degree is 1, and it is less than 1 for any angle in the interval (0,pi]. It is thus a judgment of orientation and not magnitude: two vectors with the same orientation have a cosine similarity of 1, two vectors oriented at 90 degree relative to each other have a similarity of 0, and two vectors diametrically opposed have a similarity of -1, independent of their magnitude. The cosine similarity is particularly used in positive space, where the outcome is neatly bounded in [0,1]. In other words, vectors are maximally "similar" if they&apos;re parallel and maximally "dissimilar" if they're orthogonal.

![png](common_files/cosine%20distance.png)


One advantage of cosine similarity is its low-complexity, especially for sparse vectors: only the non-zero dimensions need to be considered.

## Retrieve song features using Spotify API

To further assist with our recommendation, we are using spotify APIs to get song features.

Create a  spotify developer account and authenticate the spotify api using `clientID` and `clientSecret` for retrieving the additional features for our model. We also created a test playlist, so that we can add the recommended songs to the playlist.



```python
clientID = "a804123870274589b9ef05ae273aa8b6"
clientSecret = "a7bc0810f20d4bd287290e493488e9cd"
credentials = SpotifyClientCredentials(client_id= clientID,client_secret=clientSecret)
spotify = spotipy.Spotify(client_credentials_manager=credentials)
username= "sk08up"
scope = 'playlist-modify-public'
token = util.prompt_for_user_token(username,client_id=clientID,client_secret=clientSecret, scope=scope, 
                                   redirect_uri="https://google.com")
```


Created a function to get song info using spotipy apis. Its the similar code that we have in pre processing of MPD data. 

This one is a global function because of its generic usage for getting track info.



```python
def get_song_info(song):
    track_attributes = dict()
    local_feature = spotify.audio_features(song)[0]

    to_del = ['id','track_href','uri','analysis_url','type']
    for item in to_del:
        del(local_feature[item])

    track_attributes[song] = local_feature
    track_info = spotify.track(song) 
    artists = spotify.artist(track_info['artists'][0]['uri'])
    track_attributes[song]['artist_genres'] = artists['genres']
    track_attributes[song]['artist_popularity'] = artists['popularity']
    track_attributes[song]['explicit'] = track_info['explicit']
    track_attributes[song]['song_name'] = track_info['name']
    return track_attributes
```


## Model Overview
Our model take a new playlist or song as an input and extracts the features from spotify(via spotipy api) for each track and  provide a song for song recommendation using content based filtering. Based on the input song name, we are retrieving the basic track features such as `track_name`, `track_uri` etc. from the `Million Playlist Dataset` and additional features such as `Valence`, `Tempo`, `Loudness` etc.,. Our model tries to find the song's genre and look for matching genres(in our case, we are starting with 4 genres) to search for the similar songs. If the song is from a niche artist, we are reducing the matching criteria and again search for the similar songs until we find maximum number of recommended songs requested.

our model below is python class which has several function to assist in song recommendation. The most notable is `get_recommended_tracks` function which contains the core of our recommendation algorithm.



```python
#track_recommender recommneds songs based on a user playlist or based on randon song given by user.
class track_recommender():
    '''Create a set of recommendations'''
    
    def __init__(self,username,scope='playlist-modify-public',token=None,clientID=None,clientSecret=None,credentials=None):
        '''clientID : client ID for API app
        clientSecret: client secret for API app
        token: token for authentification
        scope: scope of endpoint requirement'''
        import pickle
        import pandas as pd
        import numpy as np
        import random
        
        self.username = username
        self.clientID = clientID
        self.clientSecret = clientSecret
        
        ## Scope
        self.scope = scope
        
        #Spotify Authentication
        #below code to get token for connecting to spotify                    
        #first we need to get all the required input to create a token from spotify. 
        #this is required in case we want to access user profile and get playlist information
        if self.clientID == None and self.clientSecret == None:
            self.credentials = credentials
        elif (self.clientID==None or self.clientSecret==None) and credentials==None:
            print ('Error')
            raise
        else:
            self.credentials = SpotifyClientCredentials(client_id= clientID,client_secret=clientSecret)
        
        # Token
        if token == None:
            self.token = util.prompt_for_user_token(self.username,client_id=self.clientID,client_secret=self.clientSecret,
                                   scope=self.scope, redirect_uri='http://localhost:8888/callback')
        else:
            self.token = token
        
    def get_user_handler(self):
        '''Creates an authenticated user instance'''
        
        self.user = spotipy.Spotify(auth=self.token,client_credentials_manager=self.credentials)
        return self.user
    
    def create_user_playlist(self,name):
        '''Creates a new playlist for user
        name: name for playlist'''
        
        self.name = name
        self.created_playlist = self.user.user_playlist_create(self.username,name=self.name)
        self.playlist_id = self.created_playlist['id']
        return self.created_playlist
    
    def get_tracks_from_playlist(self, playlist_id = None):
        '''to get list if tracks available in a playlist
        this returns the list of tracks uri for the playlist'''
        #if there is no playlist, we created a new playlist for testing
        #test_playlist is the name of the playlist for user sk08up
        if playlist_id == None:
            create_user_playlist(self,'test_playlist')
            playlist_id = self.playlist_id
        
        #getting the list of tracks from the playlist
        playlist = self.user.user_playlist(self.username,playlist_id=playlist_id)
        track_list = []
        
        for track in playlist['tracks']['items']:
            track_list.append(track['track']['uri'])
        #stored in the object variable for later use -- self.track_list
        self.track_list = track_list
        return self.track_list
        
    def get_recommended_tracks(self,max_track = 10,is_playlist = True,track_name = None, show_graph = False):
        import pickle
        import random
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import cosine_similarity
        
        # MPD Songs database - this contains the pre-processed and scaled data from Million Playlist Dataset
        mpd_full_df = pickle.load(open('./MPD_Processed_attributes.sav','rb'))

        #this function can be used for playlist as well as song, so we have to check for both scenario
        if is_playlist:
            song_recommend_df = pd.DataFrame(self.song_audio_features).T
        else:
            #based on the name of the song, we are trying to find track details
            track_uri = mpd_full_df[mpd_full_df.song_name == track_name].index.values[0]
            if track_uri:
                song_recommend_df = pd.DataFrame(get_song_info(track_uri)).T
                #setting list of songs in track 
                #for a single song, this will be blank and will fail at later point
                self.track_list = [track_uri]
            else:
                print('No Songs Found')            
                return None

        #For computational flexibility, we are shifting the following features to the end.
        df_side = song_recommend_df[['artist_popularity','explicit','artist_genres','song_name']]
        song_recommend_df.drop(labels=['artist_popularity','explicit','artist_genres','song_name'],inplace=True,axis=1)
        song_recommend_df = song_recommend_df.merge(df_side,left_index=True,right_index=True)
        song_recommend_df['key'] = song_recommend_df['key'].astype('category')
        song_recommend_df['time_signature'] = song_recommend_df['time_signature'].astype('category')
        
        # We are using the identified numerical features for calculating cosine similarity
        numeric_feature = pickle.load(open('./attribute_to_scale.sav','rb'))
        for col in song_recommend_df.columns:
            if col in numeric_feature:
                song_recommend_df[col] = pd.to_numeric(song_recommend_df[col],errors ='coerce')
        
        ## Standard Scaler
        #the scaler we have from training on MPD songs
        ss = pickle.load(open('./scaler_object.sav','rb'))
        
        song_recommend_df[['duration_ms','loudness','tempo']] = ss.transform(song_recommend_df[['duration_ms','loudness','tempo']])
        
        ## Applying multiplier
        song_recommend_df[['energy','valence']] = song_recommend_df[['energy','valence']].apply(lambda x: (x*1.5)**2)
        song_recommend_df[['tempo']] = song_recommend_df[['tempo']].apply(lambda x: (x*1.2)**2)       
        
        ## Applying multiplier to MPD songdatabase
        mpd_full_df[['energy','valence']] = mpd_full_df[['energy','valence']].apply(lambda x: (x*1.5)**2) 
        mpd_full_df[['tempo']] = mpd_full_df[['tempo']].apply(lambda x: (x*1.2)**2)
        
        ## Recommend tracks
        recommended_tracks = []
        recommended_cos_dist = []
        #since our model is an content based model, we are going to calculate based on track's numerical features
        #we are first targetting 4 artist genres to match with the songs from playlits/single songs artist genre
        artist_genre_match = 4
        while len(recommended_tracks) < max_track: 
            #if the list of songs matched is less than the max_tracks that we want to recommend, we are reducing 
            # our matching criteria to find more songs
            if(artist_genre_match < 0):
                break
            normal_track = 0
            niche_track = 0         

            # song_recommend_df is the current playlist to recommend
            for i, track in song_recommend_df.iterrows():   

                track_dist = []
                genre_tracks = []

                if len(track['artist_genres']) >=5:
                    normal_track =normal_track +1
                    for db_i,db_track in mpd_full_df.iterrows():
                        try:
                            ## Append if n genres or more intersect
                            if len(set(db_track['artist_genres']) & set(track['artist_genres']))>=artist_genre_match:
                                genre_tracks.append(db_i)
                        except:
                            continue

                else:
                    ## For niche artist
                    #if we have very less no of genre available for the track, we will look for atleast one genre                    
                    niche_track = niche_track + 1
                    for db_i,db_track in mpd_full_df.iterrows():
                        try:
                            if bool(set(db_track['artist_genres']) & set(track['artist_genres'])):
                                genre_tracks.append(db_i)
                        except:
                            continue

                #Calculating cosine similarity of the input song with the MPD song dataset.
                for i2, cluster_track in mpd_full_df[mpd_full_df.index.isin(genre_tracks)].iloc[:,:-4].iterrows():                    
                    cos_dist = cosine_similarity(track[:-4].reshape(1, -1),cluster_track.reshape(1, -1))
                    track_dist.append((cos_dist,i2))
                
                #close position starts with index 1 as the input song by which we are comparing will always be at the
                # index 0 with cosine similarity as 1                
                similar_track = sorted(track_dist,reverse=True)
                close_position=0
                if len(similar_track) > 0:
                    for data in similar_track:
                        track_uri = data[1]
                        # Check if recommended song is already inside playlist
                        if (track_uri in self.track_list or track_uri in recommended_tracks) is False:
                            recommended_tracks.append(track_uri)
                            recommended_cos_dist.append(data)
                            if len(recommended_tracks) == max_track:
                                self.recommended_tracks = recommended_tracks                                
                                break

            artist_genre_match = artist_genre_match - 1

        song_list = []
        song_data_dict = {}
        all_rec_songs = []
        all_rec_songs_df = pd.DataFrame()
        for index,track in enumerate(recommended_tracks,1):
            #getting track information based on uri
            track_obj = mpd_full_df.loc[track]
            all_rec_songs_df[track] = track_obj
            song_data_dict[track] = track_obj.song_name
            song_list.append(track_obj.song_name)
            all_rec_songs.append([index,track_obj.song_name])
        all_rec_songs_df = all_rec_songs_df.T
        all_rec_songs = np.array(all_rec_songs)
        rec_song_df = pd.DataFrame(all_rec_songs[:,1],columns=['Recommended_Songs'],index=all_rec_songs[:,0])
        display(rec_song_df)
        
        #show the similarity scores of the recommended tracks
        if show_graph:
            cosine_data = np.array([[data[0][0][0].astype(np.float64),
                                   song_data_dict[data[1]]] for data in recommended_cos_dist])
            cosine_data_df = pd.DataFrame(list(cosine_data), columns=['cosine_dist','song_name'])
            cosine_data_df = cosine_data_df.sort_values(by=['cosine_dist'], ascending=False)
            show_cosiner_sim(cosine_data_df,all_rec_songs_df)
        return song_list
    
```


Below funciton helps us visualize the cosine similarity between referenced song and recommended songs. Other half of the graph shows similarity between different key features of songs.  



```python
def show_cosiner_sim(df,all_rec_songs_df):
    df.cosine_dist = df.cosine_dist.astype(np.float64)*100
    plt.figure(figsize=(12,6))
    sns.barplot(df.song_name,df.cosine_dist,palette="Greens_d")
    plt.xticks(rotation = 75)
    plt.ylim((df.cosine_dist.min() - 5 ,100))
    
    fig,ax = plt.subplots(nrows = 2, ncols = 4,figsize = (16,8))
    features = ['danceability','energy','loudness','speechiness','acousticness','liveness','valence','tempo']
    for i in range(8):
        j = int(i/4)
        k = i%4        
        sns.barplot(np.arange(1,len(all_rec_songs_df)+1),np.abs(all_rec_songs_df[features[i]]),palette="Greens_d",ax = ax[j][k])  
        ax[j][k].set_xlabel('')
        ax[j][k].set_yticks([])
```


Below function return list of songs based on one song. show_graph parameter controls display of different visualization like cosine similarity on recommended songs and comparing recommended song features



```python
def Recommend_tracks(song_name = None,max_track = 5,show_graph = False):    
    rec_class = track_recommender(username=username,scope=scope,clientID=clientID,clientSecret=clientSecret,credentials=credentials)
    tracks = rec_class.get_recommended_tracks(max_track = max_track,is_playlist = False,track_name = song_name , show_graph = show_graph)

```


Below function returns list of songs beased on playlist id i.e list of songs available in given playlist. 



```python
def Recommend_tracks_by_playlist(pl_uri,add_tracks='max',max_track = 5):
    uri = pl_uri.split(':')[-1]    
    rec_class = track_recommender(username=username,scope=scope,clientID=clientID,clientSecret=clientSecret,credentials=credentials)
    rec_class.get_user_handler()
    rec_class.get_tracks_from_playlist(uri)
    rec_class.user_get_playlist_track_audio_feat()
    tracks = rec_class.get_recommended_tracks(max_track = max_track)

```

