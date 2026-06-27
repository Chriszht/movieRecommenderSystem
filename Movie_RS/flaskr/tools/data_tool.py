import os
import time
import pandas as pd


def loadData():
    return getMovies(), getGenre(), getRates()


# movieId,title,year,overview,cover_url,genres
def getMovies():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/movie_info.csv"
    df = pd.read_csv(path)
    df['genres'] = df.genres.str.split('|')
    df['overview'] = df['overview'].fillna('')
    # Derived column used by the UI template (fixes the original release_date bug).
    df['release_date'] = df['year'].apply(
        lambda y: '' if pd.isna(y) else str(int(y))
    )
    return df


# A list of the genres.
def getGenre():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/genre.csv"
    df = pd.read_csv(path, delimiter="|", names=["name", "id"])
    df.set_index('id')
    return df


# user id, item id, rating, timestamp
# Enhanced: keep timestamp so time-aware algorithms can use it.
def getRates(keep_timestamp=True):
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/ratings.csv"
    df = pd.read_csv(path, delimiter=",", header=0,
                     names=["userId", "movieId", "rating", "timestamp"])
    if keep_timestamp:
        return df[['userId', 'movieId', 'rating', 'timestamp']]
    return df[['userId', 'movieId', 'rating']]


# itemID | userID | rating | timestamp
# The front-end encodes each rating as "userId|movieId|rating|timestamp".
# If timestamp is missing or 0, we fall back to current time.
def ratesFromUser(rates, keep_timestamp=True):
    itemID, userID, rating, ts = [], [], [], []
    now = int(time.time())

    for rate in rates:
        items = rate.split("|")
        userID.append(int(items[0]))
        itemID.append(int(items[1]))
        rating.append(int(items[2]))
        if len(items) >= 4:
            try:
                t = int(items[3])
            except ValueError:
                t = 0
            ts.append(t if t > 0 else now)
        else:
            ts.append(now)

    ratings_dict = {
        "userId": userID,
        "movieId": itemID,
        "rating": rating,
    }
    if keep_timestamp:
        ratings_dict["timestamp"] = ts

    return pd.DataFrame(ratings_dict)