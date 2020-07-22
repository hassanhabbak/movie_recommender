# Movie Recommender
> This is a movie recommender implementation of user to movie and movie to movie recommendations. The methods used are primarily focused on embeddings to extrapolate similarity between users and items.


## Dependencies

`conda create --name <env> --file requirements.txt`

## How to use

## movies_metadata module

Loading in the meta data features:

- Cleans data from duplicates
- Convert adult tag on movies to bool
- Label Encodes genres after cleanning
- Drops incorrect Iso for languages
- Gets numerical features and corrects wrong values
- Bucketizes the decade the movie was launched in
- Creates a flag of whether the movie is recent or not
- NLP processing of overview description (TFIDF + LDA)

```python
meta_df = pd.read_csv('data/movies_metadata.csv')
movies_df = mmd.get_movie_features(meta_df)
movies_df.head()
```

    C:\Users\hhapp\anaconda3\envs\tensorflow\lib\site-packages\IPython\core\interactiveshell.py:3063: DtypeWarning: Columns (10) have mixed types.Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)





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
      <th>adult</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>runtime</th>
      <th>popularity</th>
      <th>decade_label</th>
      <th>released_recently</th>
      <th>G_0</th>
      <th>G_1</th>
      <th>G_2</th>
      <th>...</th>
      <th>G_10</th>
      <th>G_11</th>
      <th>G_12</th>
      <th>G_13</th>
      <th>G_14</th>
      <th>G_15</th>
      <th>G_16</th>
      <th>G_17</th>
      <th>G_18</th>
      <th>G_19</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>862</th>
      <td>False</td>
      <td>5415.0</td>
      <td>7.7</td>
      <td>81.0</td>
      <td>21.946943</td>
      <td>2.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8844</th>
      <td>False</td>
      <td>2413.0</td>
      <td>6.9</td>
      <td>104.0</td>
      <td>17.015539</td>
      <td>2.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15602</th>
      <td>False</td>
      <td>92.0</td>
      <td>6.5</td>
      <td>101.0</td>
      <td>11.712900</td>
      <td>2.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>31357</th>
      <td>False</td>
      <td>34.0</td>
      <td>6.1</td>
      <td>127.0</td>
      <td>3.859495</td>
      <td>2.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11862</th>
      <td>False</td>
      <td>173.0</td>
      <td>5.7</td>
      <td>106.0</td>
      <td>8.387519</td>
      <td>2.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



## users module

Collaborative filtering is a technique that usually relies on matrix factorization to reduce a fat matrix to two thin ones, horizontally for user similarity, and vertically for item similarity. Using matrix completition, we could deduce how likely one user is to rate another. However the dataset of user iteractions is large and can become bigger by time. This is why I selected NN approach.

In this approach, a network is constructed to have two embedding layers learned against the movie rating as a target. One layer is for the user, the other is for the movie. The network weights for the layer is optimized to reduce the error in predicting the rating. The vectors from the embedding layers then will be the vector representation of similarity of movies and users.

#### Using NN collaborative filtering

In this part, the users and movies are represented as label encoded input to the network. Only the rating behavior is then the deciding factor for helping the model converge on a solution.

```python
ratings_df = pd.read_csv('data/ratings.csv')
df, user_le, movie_le = users.add_labels(ratings_df)
df.head()
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>user_label</th>
      <th>movie_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>81834</td>
      <td>5.0</td>
      <td>1425942133</td>
      <td>0</td>
      <td>16196</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>112552</td>
      <td>5.0</td>
      <td>1425941336</td>
      <td>0</td>
      <td>23638</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>98809</td>
      <td>0.5</td>
      <td>1425942640</td>
      <td>0</td>
      <td>20011</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>99114</td>
      <td>4.0</td>
      <td>1425941667</td>
      <td>0</td>
      <td>20089</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>858</td>
      <td>5.0</td>
      <td>1425941523</td>
      <td>0</td>
      <td>843</td>
    </tr>
  </tbody>
</table>
</div>



```python
movie_train, movie_val, movie_test, user_train, \
user_val, user_test, rating_train, rating_val, \
rating_test = users.create_training_data(df.movie_label.values, df.user_label.values, df.rating)
```

```python
model, history = users.train_nn_user_behaviour(df, movie_train, movie_val, user_train, user_val, rating_train, rating_val)
```

To extract the embedding layers:

```python
movie_vec = users.extract_weights('movie_vec', model)
user_vec = users.extract_weights('user_vec', model)
```

And to produce the output, use:

```python
users.get_user_movie_output(model, eval_df, user_le, movie_le)
```

#### Using content similarity and collaborative embeddings

For this part, the movies are now represented as a dimenstionally reduced vector of the metadata features for the movie. This is combined with the movie embeddings from the previous NN to have a representation of both content similarity and user behaviour similarity. Once combined, I apply UMAP on top to reduce the dimensionality and construct cosine similarity matrix that will have the distance between -1 to 1 for all movies and each other.

```python
movie_to_movie_df = get_movie_to_movie_rating(model, movie_le, embedding_df)
movie_to_movie_df.to_csv('output/movie_to_movie.csv', index=False)
```
