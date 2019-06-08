---
nav_include: 2
title: Pre Processing
notebook: Pre Processing.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}

## Retrieve list of Track URIs 
For recommending songs, we would require few more attibutes from songs. <br> These additional attributes will be retrieved using spotify apis. <br>So first we will store all the songs uri and later use that uri to retrieve information from spotify. <br>Track and songs wil be used interchangingly as both are refer to same thing



```python
def process_mpd(path):
    # this method retrieves list of track uris
    
    count = 0
    filenames = os.listdir(path)
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            #fetching files that end with .json
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            #loading json files for getting track uris
            for playlist in mpd_slice['playlists']:                
                #data_playlists.append([playlist[col] for col in playlist_col])
                for track in playlist['tracks']:
                    track_uris.append(track.get('track_uri'))   
                    count = count +1 
    #returnig list of track uris
    return track_uris

```


Let&apos;s retrieve and store list of Tracks



```python
#getting list of track uris
track_uris = []
path = 'D:/Machine Learning/Data Science/final project/mpd/data'
track_uris = process_mpd(path)

#storing in a file that can be used later for retrieving track information
with open("MPD_songlist.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(songlist)


```


We are storing files in between as the data is quite huge and if there is any issue with kernel or any other failure, we would lose lot of time, so we started to store data at certain points.
Also this aproach helped us as we do not have to start from pre processing again as reading those json files took lot of time.<br> As the main recommnder file will need some preprocessed data and keeping in a local file helped.

## Reading data from the saved list of track file



```python
with open('MPD_songlist.csv', 'r') as f:
    reader = csv.reader(f)
    mpd_song_list = list(reader)

```


22.6 million songs, these are the total no of songs retreived from MPD 



```python
#list of track uris
#these are the total no of songs retreived from MPD
#22.6 million songs
mpd_song_list = mpd_song_list[0]
len(mpd_song_list)
```





    2262292



## Retrieving audio feature using Spotify API

Based on these uris, we are going to retrieve information from spotify. attributes retreived from spotify will be used for recommedations. Attributes like artist_genres, acousticness, tempo,valence and energy to name a few.<br>
Cosine similarity is being used to get the list of recommended songs.

### Spotify API authentication



```python
#for retrieval of information from spotify
#Authentication for the Spotify api

clientID = 'c3c5fc22422a4d4e838e80966b832f20'
clientSecret = '5a69ae12ffff48f190ec2e90eae1ad42'
credentials = SpotifyClientCredentials(client_id= clientID,client_secret=clientSecret)
spotify = spotipy.Spotify(client_credentials_manager=credentials)
```


### Call 'audio_features' API



```python
 spotify.audio_features('spotify:track:14J3PO0VnhtcRa31r7Aj1L')

```





    [{'acousticness': 0.0975,
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/14J3PO0VnhtcRa31r7Aj1L',
      'danceability': 0.507,
      'duration_ms': 273573,
      'energy': 0.71,
      'id': '14J3PO0VnhtcRa31r7Aj1L',
      'instrumentalness': 1.08e-05,
      'key': 5,
      'liveness': 0.17,
      'loudness': -5.02,
      'mode': 1,
      'speechiness': 0.0291,
      'tempo': 134.647,
      'time_signature': 4,
      'track_href': 'https://api.spotify.com/v1/tracks/14J3PO0VnhtcRa31r7Aj1L',
      'type': 'audio_features',
      'uri': 'spotify:track:14J3PO0VnhtcRa31r7Aj1L',
      'valence': 0.45}]



### Save as SAV files

The below code will save all the tracks attribute in .sav file. we can use this file at a later point for recommendation. Saving these files helped us getting all the attributes of MPD song list. we had issue earlier while running for MPD data as this ran for multiple days and any failure caused lot of time wastage. So it was computationally very expenive process, so had to use .sav file to store locally.

Note: spotify.audio_features API can take a maximum of 50 uris as input, so we are incrementing our counter with 50.



```python
i = 0 ;
track_attributes= {}
failed_song = []
while i < len(mpd_song_list):
    #spotify.audio_features can take a maximum of 50 uris as input, so we are incrementing our counter to 50.
    # and we wanted to make least call to spotify as it was taking lot of time to get data for all MPD tracks
    
    k = i+50    
    if k > len(mpd_song_list):
        #there are lot of failures so our final output may not be a multiple of 50
        k = len(mpd_song_list)
    
    #temporary list to store those 50 songs
    temp_song_list = mpd_song_list[i:k]
    
    #we want to keep track of all failed songs.
    failed_track = None
    count = 0
    
    try:
        #spotify.tracks will retrun a dictionary with key = tracks. From there we can get list of tracks
        track_info = spotify.tracks(temp_song_list, market=None)  
        track_info = track_info['tracks']
        # getting track data in a list for creating a dataframe, wehere we can use track uris to retrieve information.
        # this will help to improve performance.
        temp_track_data = [[data['uri'],data['artists'][0]['uri'],data['name'],data['explicit']] for data in track_info]
        # extracting song name as our start point for our recommendation is 
        mpd_df = pd.DataFrame(temp_track_data,columns=['uri','artists','song_name','explicit'])
        mpd_df.set_index('uri',inplace=True)
        # getting attributes for tracks and these are the main attributes that will be used for recomendation.
        audio_features = spotify.audio_features(temp_song_list)
    #         print('step1')
        for audio_feature in tqdm(audio_features):            
            track_uri = audio_feature['uri']
            failed_track = track_uri
        ## Access spotify api to retrieve audio features for specific track    
            track_attributes[track_uri] = audio_feature
            track_data = mpd_df.loc[track_uri]
            # getting artist information from dictionary
            artist_info = spotify.artist(track_data.artists)
            track_attributes[track_uri]['artist_genres'] = artist_info['genres'] #artist_info[0]
            track_attributes[track_uri]['artist_popularity'] = artist_info['popularity'] #artist_info[1]
            track_attributes[track_uri]['explicit'] = track_data.explicit
            track_attributes[track_uri]['song_name'] = track_data.song_name
            i+=1
            #save data when i reaches multiple of 10000
            if i%10000 == 0:
                #as mentioned, we are storing data at certain interval, so that we dont have to run from start and save time.                
                print('Saving Songs :',i)
                filename = 'MPD_songs_{}.sav'.format(i)
                pickle.dump(track_attributes,open(filename,'wb'))
                #initiaze our object
                track_attributes= {}
    #         i+=40
        print('no of songs done:',i)
            
    except:
        print('failed:',i)
        i = i+1
        failed_song.append(failed_song)
        continue
```


## Preprocess the updated Data

### Retreiving MPD tracks and their attributes

Since MPD is quite huge(around 22.6 million songs), we did split our pre processing in two files. One where we can store all .SAV file and other where we can retrieve those file and get the final verison of our data.



```python
mpd_song_attributes = dict()

i=10000
while i <=2263000:
    try:
    #     print('opening ' + 'MPD_songs_{}.sav'.format(i))
        a = pickle.load(open('MPD_songs_{}.sav'.format(i),'rb'))
        mpd_song_attributes.update(a)
        i += 10000
    except:
        i += 10000
        continue
```


### Remove some insignificant attributes

Let's create a dataframe to store and remove some of the attribute that does not have significance in terms of prediction



```python
#created a dataframe for computations
mpd_df = pd.DataFrame(mpd_song_attributes)
mpd_df = mpd_df.T
#removing some of the attribute that does not have significance in terms of prediction
#some of the informational data
mpd_df.drop(labels=['analysis_url','id','track_href','type','uri'],axis=1,inplace=True)
```




```python
mpd_df.head()
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
      <th>acousticness</th>
      <th>artist_genres</th>
      <th>artist_popularity</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>explicit</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>song_name</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>spotify:track:000Cr0YJII1jkMBHoiti4p</th>
      <td>0.783</td>
      <td>[]</td>
      <td>19</td>
      <td>0.698</td>
      <td>39133</td>
      <td>0.834</td>
      <td>True</td>
      <td>0</td>
      <td>5</td>
      <td>0.959</td>
      <td>-6.952</td>
      <td>0</td>
      <td>CA is The Flip Flop Capitol</td>
      <td>0.916</td>
      <td>50.558</td>
      <td>4</td>
      <td>0.458</td>
    </tr>
    <tr>
      <th>spotify:track:000DfZJww8KiixTKuk9usJ</th>
      <td>0.366</td>
      <td>[reggae rock]</td>
      <td>48</td>
      <td>0.631</td>
      <td>357573</td>
      <td>0.513</td>
      <td>False</td>
      <td>4.16e-06</td>
      <td>2</td>
      <td>0.109</td>
      <td>-6.376</td>
      <td>1</td>
      <td>Earthlings</td>
      <td>0.0293</td>
      <td>120.365</td>
      <td>4</td>
      <td>0.307</td>
    </tr>
    <tr>
      <th>spotify:track:000I8hOWZ1T94MdmsFpt5P</th>
      <td>0.823</td>
      <td>[]</td>
      <td>37</td>
      <td>0.523</td>
      <td>331640</td>
      <td>0.176</td>
      <td>False</td>
      <td>0.223</td>
      <td>10</td>
      <td>0.131</td>
      <td>-13.782</td>
      <td>1</td>
      <td>Not Nice</td>
      <td>0.0287</td>
      <td>52.995</td>
      <td>4</td>
      <td>0.0526</td>
    </tr>
    <tr>
      <th>spotify:track:000q6cu9IRk2Ypfwb8671l</th>
      <td>0.946</td>
      <td>[adult standards, big band, cabaret, christmas...</td>
      <td>33</td>
      <td>0.513</td>
      <td>86653</td>
      <td>0.0846</td>
      <td>False</td>
      <td>0.684</td>
      <td>9</td>
      <td>0.115</td>
      <td>-17.235</td>
      <td>0</td>
      <td>So in Love</td>
      <td>0.078</td>
      <td>168.347</td>
      <td>5</td>
      <td>0.573</td>
    </tr>
    <tr>
      <th>spotify:track:000svt0m4OSEC2RfZL9j0J</th>
      <td>0.0105</td>
      <td>[aggrotech, dark wave, ebm, electro-industrial...</td>
      <td>40</td>
      <td>0.488</td>
      <td>245082</td>
      <td>0.947</td>
      <td>False</td>
      <td>0.244</td>
      <td>7</td>
      <td>0.096</td>
      <td>-6.404</td>
      <td>1</td>
      <td>Mein Weg</td>
      <td>0.0528</td>
      <td>139.956</td>
      <td>4</td>
      <td>0.61</td>
    </tr>
  </tbody>
</table>
</div>





```python
#saving the updated data in .sav file
pickle.dump(mpd_df, open('MPD_songs_final.sav','wb'))
```


### Finalize pre processing

We will here load the final preprocessed dataset. As there are few columns that won't be required for computation,
we will be keeping that on the right side of the data frame for easy calculation as those columns can be filtered out later.



```python
from sklearn.preprocessing import StandardScaler
```




```python
mpd_df = pickle.load(open('MPD_songs_final.sav','rb'))

mpd_df_side = mpd_df[['artist_popularity','explicit','artist_genres','song_name']]

mpd_df.drop(labels=['artist_popularity','explicit','artist_genres','song_name'],inplace=True,axis=1)
mpd_df = mpd_df.merge(mpd_df_side,left_index=True,right_index=True)
mpd_df.head()
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
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
      <th>artist_popularity</th>
      <th>explicit</th>
      <th>artist_genres</th>
      <th>song_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>spotify:track:000Cr0YJII1jkMBHoiti4p</th>
      <td>0.783</td>
      <td>0.698</td>
      <td>39133</td>
      <td>0.834</td>
      <td>0</td>
      <td>5</td>
      <td>0.959</td>
      <td>-6.952</td>
      <td>0</td>
      <td>0.916</td>
      <td>50.558</td>
      <td>4</td>
      <td>0.458</td>
      <td>19</td>
      <td>True</td>
      <td>[]</td>
      <td>CA is The Flip Flop Capitol</td>
    </tr>
    <tr>
      <th>spotify:track:000DfZJww8KiixTKuk9usJ</th>
      <td>0.366</td>
      <td>0.631</td>
      <td>357573</td>
      <td>0.513</td>
      <td>4.16e-06</td>
      <td>2</td>
      <td>0.109</td>
      <td>-6.376</td>
      <td>1</td>
      <td>0.0293</td>
      <td>120.365</td>
      <td>4</td>
      <td>0.307</td>
      <td>48</td>
      <td>False</td>
      <td>[reggae rock]</td>
      <td>Earthlings</td>
    </tr>
    <tr>
      <th>spotify:track:000I8hOWZ1T94MdmsFpt5P</th>
      <td>0.823</td>
      <td>0.523</td>
      <td>331640</td>
      <td>0.176</td>
      <td>0.223</td>
      <td>10</td>
      <td>0.131</td>
      <td>-13.782</td>
      <td>1</td>
      <td>0.0287</td>
      <td>52.995</td>
      <td>4</td>
      <td>0.0526</td>
      <td>37</td>
      <td>False</td>
      <td>[]</td>
      <td>Not Nice</td>
    </tr>
    <tr>
      <th>spotify:track:000q6cu9IRk2Ypfwb8671l</th>
      <td>0.946</td>
      <td>0.513</td>
      <td>86653</td>
      <td>0.0846</td>
      <td>0.684</td>
      <td>9</td>
      <td>0.115</td>
      <td>-17.235</td>
      <td>0</td>
      <td>0.078</td>
      <td>168.347</td>
      <td>5</td>
      <td>0.573</td>
      <td>33</td>
      <td>False</td>
      <td>[adult standards, big band, cabaret, christmas...</td>
      <td>So in Love</td>
    </tr>
    <tr>
      <th>spotify:track:000svt0m4OSEC2RfZL9j0J</th>
      <td>0.0105</td>
      <td>0.488</td>
      <td>245082</td>
      <td>0.947</td>
      <td>0.244</td>
      <td>7</td>
      <td>0.096</td>
      <td>-6.404</td>
      <td>1</td>
      <td>0.0528</td>
      <td>139.956</td>
      <td>4</td>
      <td>0.61</td>
      <td>40</td>
      <td>False</td>
      <td>[aggrotech, dark wave, ebm, electro-industrial...</td>
      <td>Mein Weg</td>
    </tr>
  </tbody>
</table>
</div>



Below we are removing any column with no value and since the range of values of some of the columns ('duration_ms','loudness','tempo') data varies widely we are standardizing the features to use in our recommendation model. Finally saving the pre-processed data into a 'MPD_Processed_attributes.sav' file.



```python
mpd_df['key'] = mpd_df['key'].astype('category')
mpd_df['time_signature'] = mpd_df['time_signature'].astype('category')

non_numeric = ['key','mode','time_signature','artist_popularity','explicit','artist_genres','song_name']

for col in mpd_df.columns:
    if col not in non_numeric:
        mpd_df[col] = pd.to_numeric(mpd_df[col],errors ='coerce')

#removing any column with no value
mpd_df.dropna(inplace=True)

#Since the range of values of some of the columns ('duration_ms','loudness','tempo') data varies widely
ss = StandardScaler()
scaler = ss.fit(mpd_df[['duration_ms','loudness','tempo']])

mpd_df[['duration_ms','loudness','tempo']] = scaler.transform(mpd_df[['duration_ms','loudness','tempo']])

scale_features = [col for col in mpd_df.columns if col not in non_numeric]

pickle.dump(scaler,open('scaler_object.sav','wb'))

#feature to scale, same used during preduction
pickle.dump(scale_features,open('attribute_to_scale.sav','wb'))

#list of total no of scaled songs from MPD
pickle.dump(mpd_df,open('MPD_Processed_attributes.sav','wb'))
mpd_df.head()
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
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
      <th>artist_popularity</th>
      <th>explicit</th>
      <th>artist_genres</th>
      <th>song_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>spotify:track:000Cr0YJII1jkMBHoiti4p</th>
      <td>0.7830</td>
      <td>0.698</td>
      <td>-1.326765</td>
      <td>0.8340</td>
      <td>0.000000</td>
      <td>5</td>
      <td>0.959</td>
      <td>0.482274</td>
      <td>0</td>
      <td>0.9160</td>
      <td>-2.326227</td>
      <td>4</td>
      <td>0.4580</td>
      <td>19</td>
      <td>True</td>
      <td>[]</td>
      <td>CA is The Flip Flop Capitol</td>
    </tr>
    <tr>
      <th>spotify:track:000DfZJww8KiixTKuk9usJ</th>
      <td>0.3660</td>
      <td>0.631</td>
      <td>0.700937</td>
      <td>0.5130</td>
      <td>0.000004</td>
      <td>2</td>
      <td>0.109</td>
      <td>0.584651</td>
      <td>1</td>
      <td>0.0293</td>
      <td>0.013636</td>
      <td>4</td>
      <td>0.3070</td>
      <td>48</td>
      <td>False</td>
      <td>[reggae rock]</td>
      <td>Earthlings</td>
    </tr>
    <tr>
      <th>spotify:track:000I8hOWZ1T94MdmsFpt5P</th>
      <td>0.8230</td>
      <td>0.523</td>
      <td>0.535805</td>
      <td>0.1760</td>
      <td>0.223000</td>
      <td>10</td>
      <td>0.131</td>
      <td>-0.731677</td>
      <td>1</td>
      <td>0.0287</td>
      <td>-2.244541</td>
      <td>4</td>
      <td>0.0526</td>
      <td>37</td>
      <td>False</td>
      <td>[]</td>
      <td>Not Nice</td>
    </tr>
    <tr>
      <th>spotify:track:000q6cu9IRk2Ypfwb8671l</th>
      <td>0.9460</td>
      <td>0.513</td>
      <td>-1.024176</td>
      <td>0.0846</td>
      <td>0.684000</td>
      <td>9</td>
      <td>0.115</td>
      <td>-1.345406</td>
      <td>0</td>
      <td>0.0780</td>
      <td>1.621946</td>
      <td>5</td>
      <td>0.5730</td>
      <td>33</td>
      <td>False</td>
      <td>[adult standards, big band, cabaret, christmas...</td>
      <td>So in Love</td>
    </tr>
    <tr>
      <th>spotify:track:000svt0m4OSEC2RfZL9j0J</th>
      <td>0.0105</td>
      <td>0.488</td>
      <td>-0.015362</td>
      <td>0.9470</td>
      <td>0.244000</td>
      <td>7</td>
      <td>0.096</td>
      <td>0.579674</td>
      <td>1</td>
      <td>0.0528</td>
      <td>0.670307</td>
      <td>4</td>
      <td>0.6100</td>
      <td>40</td>
      <td>False</td>
      <td>[aggrotech, dark wave, ebm, electro-industrial...</td>
      <td>Mein Weg</td>
    </tr>
  </tbody>
</table>
</div>


