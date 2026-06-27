import os

import numpy as np
from flask import (
    Blueprint, render_template, request, jsonify
)

from .tools.data_tool import *
from .tools import recommenders as rec

from surprise import Reader
from surprise import KNNBasic, KNNWithMeans
from surprise import Dataset
from sklearn.metrics.pairwise import cosine_similarity

bp = Blueprint('main', __name__, url_prefix='/')

# ---------------------------------------------------------------------------
# Algorithm / UI switches for the A/B test.
#   REC_ALGO  : "original" (demo kNN+multi-hot) | "enhanced" (hybrid+MMR)
#   UI_MODE   : "original" | "enhanced"
# Set via environment variables, e.g. (PowerShell):
#       $env:REC_ALGO="enhanced"; $env:UI_MODE="enhanced"; flask --app flaskr run
# ---------------------------------------------------------------------------
REC_ALGO = os.environ.get('REC_ALGO', 'enhanced').lower()
UI_MODE = os.environ.get('UI_MODE', 'enhanced').lower()

movies, genres, rates = loadData()
# Pre-compute the TF-IDF + genre item matrix once at start-up (expensive).
ITEM_MATRIX, _TFIDF, _GENRE_LIST = rec.build_content_matrix(movies)


def _compute_trending():
    """Popularity-times-quality ranking, used for the Trending Now strip."""
    agg = (rates.groupby('movieId')
                .agg(count=('rating', 'count'), mean=('rating', 'mean')))
    agg = agg[agg['count'] >= 20]
    agg['score'] = agg['mean'] * np.log1p(agg['count'])
    known = set(movies['movieId'].tolist())
    ranked = [int(mid) for mid in agg.sort_values('score', ascending=False).index
              if int(mid) in known]
    return ranked[:30]


TRENDING_IDS = _compute_trending()


PROFILE_IDS = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06']


def _current_profile():
    """Active profile prefix (e.g. 'P01'). Empty string = legacy/no profile."""
    pid = (request.cookies.get('profile_id') or '').strip()
    return pid if pid in PROFILE_IDS else ''


def _pp(name):
    """Prefix a cookie name with the current profile id (if any)."""
    p = _current_profile()
    return f'{p}_{name}' if p else name


@bp.route('/', methods=('GET', 'POST'))
def index():
    default_genres = genres.to_dict('records')
    user_genres = _cookie_list('user_genres')
    user_rates = _cookie_list('user_rates')
    user_likes = _cookie_list('user_likes')
    user_dislikes = _cookie_list('user_dislikes')
    user_interests = _cookie_list('user_interests')

    default_genres_movies = getMoviesByGenres(user_genres)[:10]
    trending_movies = _movies_in_order(TRENDING_IDS)[:12]

    if REC_ALGO == 'original':
        recs, rec_msg = getRecommendationBy(user_rates)
        likes_similars, likes_msg = getLikedSimilarBy(
            [int(x) for x in user_likes]
        )
        explanations = {}
    else:
        recs, likes_similars, rec_msg, likes_msg, explanations = \
            enhanced_recommendations(
                user_rates, user_likes, user_dislikes, user_interests,
            )

    likes_movies = getUserLikesBy(user_likes)
    dislikes_movies = getUserLikesBy(user_dislikes)

    template_name = 'index.html' if UI_MODE == 'enhanced' else 'index_original.html'
    return render_template(
        template_name,
        genres=default_genres,
        user_genres=user_genres,
        user_rates=user_rates,
        user_likes=user_likes,
        user_dislikes=user_dislikes,
        user_interests=user_interests,
        default_genres_movies=default_genres_movies,
        trending_movies=trending_movies,
        recommendations=recs,
        recommendations_message=rec_msg,
        likes_similars=likes_similars,
        likes_similar_message=likes_msg,
        likes=likes_movies,
        dislikes=dislikes_movies,
        explanations=explanations,
        rec_algo=REC_ALGO,
        ui_mode=UI_MODE,
        profile_id=_current_profile(),
        profile_ids=PROFILE_IDS,
    )


def _cookie_list(name):
    raw = request.cookies.get(_pp(name))
    return raw.split(',') if raw else []


def getUserLikesBy(user_likes):
    results = []

    if len(user_likes) > 0:
        mask = movies['movieId'].isin([int(movieId) for movieId in user_likes])
        results = movies.loc[mask]

        original_orders = pd.DataFrame()
        for _id in user_likes:
            movie = results.loc[results['movieId'] == int(_id)]
            if len(original_orders) == 0:
                original_orders = movie
            else:
                original_orders = pd.concat([movie, original_orders])
        results = original_orders

    if len(results) > 0:
        return results.to_dict('records')
    return results

def is_genre_match(movie_genres, interested_genres):
    return bool(set(movie_genres).intersection(set(interested_genres)))

def getMoviesByGenres(user_genres):
    results = []
    if len(user_genres) > 0:
        genres_mask = genres['id'].isin([int(id) for id in user_genres])
        user_genres = [1 if has is True else 0 for has in genres_mask]
        user_genres_df = pd.DataFrame(user_genres,columns=['value'])
        user_genres_df = pd.concat([user_genres_df, genres['name']], axis=1)
        interested_genres = user_genres_df[user_genres_df['value'] == 1]['name'].tolist()
        results = movies[movies['genres'].apply(lambda x: is_genre_match(x, interested_genres))]

    if len(results) > 0:
        return results.to_dict('records')
    return results

# Baseline (original demo) user-based kNN — kept for A/B comparison.
def getRecommendationBy(user_rates):
    results = []
    if len(user_rates) > 0:
        reader = Reader(rating_scale=(1, 5))
        algo = KNNWithMeans(sim_options={'name': 'pearson', 'user_based': True})
        user_rates = ratesFromUser(user_rates, keep_timestamp=False)
        training_rates = pd.concat(
            [rates[['userId', 'movieId', 'rating']], user_rates],
            ignore_index=True,
        )
        training_data = Dataset.load_from_df(training_rates, reader=reader)
        trainset = training_data.build_full_trainset()
        algo.fit(trainset)
        all_movie_ids = movies['movieId'].unique()
        user_id = 611
        rated_movie_ids = user_rates[user_rates['userId'] == user_id]['movieId'].tolist()
        predictions = [algo.predict(user_id, movie_id) for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
        top_predictions = [pred for pred in predictions]
        # sort predicted ratings in a descending order
        top_predictions.sort(key=lambda x: x.est, reverse=True)
        # Select the top-K items (e.g., 12)
        top_movie_ids = [pred.iid for pred in top_predictions[:12]]
        results = movies[movies['movieId'].isin(top_movie_ids)]


    # Return the result
    if len(results) > 0:
        return results.to_dict('records'), "These movies are recommended based on your ratings."
    return results, "No recommendations."



# Modify this function
def getLikedSimilarBy(user_likes):
    results = []
    if len(user_likes) > 0:
        # Step 1: Representing items with multi-hot vectors
        item_rep_matrix, item_rep_vector, feature_list = item_representation_based_movie_genres(movies)
        # Step 2: Building user profile
        user_profile = build_user_profile(user_likes, item_rep_vector, feature_list)
        # Step 3: Predicting user interest in items
        results = generate_recommendation_results(user_profile, item_rep_matrix, item_rep_vector, 12)
    if len(results) > 0:
        return results.to_dict('records'), "The movies are similar to your liked movies."
    return results, "No similar movies found."


# Step 1: Representing items with multi-hot vectors
def item_representation_based_movie_genres(movies_df):
    movies_with_genres = movies_df.copy(deep=True)
    genre_list = []
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            movies_with_genres.at[index, genre] = 1
            if genre not in genre_list:
                genre_list.append(genre)

    movies_with_genres = movies_with_genres.fillna(0)

    movies_genre_matrix = movies_with_genres[genre_list].to_numpy()

    return movies_genre_matrix, movies_with_genres, genre_list

# Step 2: Building user profile
def build_user_profile(movieIds, item_rep_vector, feature_list, weighted=True, normalized=True):
    user_movie_rating_df = item_rep_vector[item_rep_vector['movieId'].isin(movieIds)]
    user_movie_df = user_movie_rating_df[feature_list].mean()
    user_profile = user_movie_df.T

    if normalized:
        user_profile = user_profile / sum(user_profile.values)

    return user_profile
# Step 3: Predicting user preference for items
def generate_recommendation_results(user_profile,item_rep_matrix, movies_data, k=12):
    u_v = user_profile.values
    u_v_matrix =  [u_v]
    recommendation_table =  cosine_similarity(u_v_matrix,item_rep_matrix)
    recommendation_table_df = movies_data.copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]
    rec_result = recommendation_table_df.sort_values(by=['similarity'], ascending=False)[:k]
    return rec_result



# ===========================================================================
# Enhanced pipeline (A1 + A2 + A3 + MMR + explanations)
# ===========================================================================
def _parse_interests(raw_list):
    """Each entry is 'movieId|status' where status in {'int','not','skip'}."""
    interested, not_interested, skipped = [], [], []
    for e in raw_list or []:
        parts = e.split('|')
        if len(parts) != 2:
            continue
        try:
            mid = int(parts[0])
        except ValueError:
            continue
        s = parts[1]
        if s == 'int':
            interested.append(mid)
        elif s == 'not':
            not_interested.append(mid)
        elif s == 'skip':
            skipped.append(mid)
    return interested, not_interested, skipped


def enhanced_recommendations(user_rates_raw, user_likes_raw,
                             user_dislikes_raw, user_interests_raw=None):
    """
    Run the time-decayed kNN + TF-IDF content model, fuse them, diversify,
    and attach per-item explanations. Returns (main_list, similar_list,
    main_msg, similar_msg, explanations_map).
    """
    if len(user_rates_raw) == 0 and not user_likes_raw and not user_interests_raw:
        return [], [], "No recommendations.", "No similar movies found.", {}

    user_rates_df = ratesFromUser(user_rates_raw, keep_timestamp=True) \
        if user_rates_raw else pd.DataFrame(
            columns=['userId', 'movieId', 'rating', 'timestamp'])
    liked_ids = [int(x) for x in user_likes_raw] if user_likes_raw else []
    disliked_ids = [int(x) for x in user_dislikes_raw] if user_dislikes_raw else []
    interested_ids, not_interested_ids, skipped_ids = \
        _parse_interests(user_interests_raw)
    user_id = 611
    all_movie_ids = movies['movieId'].unique().tolist()

    # ---- A1. Time-decayed CF ----
    if len(user_rates_df) > 0:
        all_rates_with_ts = rates[['userId', 'movieId', 'rating', 'timestamp']]
        cf_top_ids, cf_pred_map, _neighbors = rec.time_decayed_knn(
            all_rates_with_ts, user_rates_df, all_movie_ids,
            user_id=user_id, top_k=60,
        )
    else:
        cf_pred_map = {}

    # ---- A2. Content-based on TF-IDF(overview) + genres ----
    rated_map = {
        int(r.movieId): float(r.rating) / 5.0
        for r in user_rates_df.itertuples() if r.rating >= 3
    }
    # "Interested but not seen" signals feed the content profile at half
    # weight of an explicit like (they indicate taste, not experience).
    cb_scores_vec = rec.content_based_scores(
        movies, ITEM_MATRIX, liked_ids,
        rated_ids_with_weight=rated_map,
        interested_ids=interested_ids,
    )

    # Align CF scores to the full movie index.
    cf_scores_vec = np.zeros(len(movies))
    for i, mid in enumerate(movies['movieId'].tolist()):
        if mid in cf_pred_map:
            cf_scores_vec[i] = cf_pred_map[mid]

    # ---- A3. Hybrid fusion ----
    hybrid = rec.hybrid_scores(cf_scores_vec, cb_scores_vec, alpha=0.6)
    # "Not interested" soft-demotes similar items (less severe than a full
    # dislike). We also compute a penalty from their content vectors.
    if not_interested_ids:
        neg = rec.content_based_scores(
            movies, ITEM_MATRIX, not_interested_ids,
        )
        hybrid = hybrid - 0.3 * np.asarray(neg).ravel()
    # Hard exclude: already-rated, disliked, and explicitly-skipped titles.
    rated_ids = set(user_rates_df['movieId'].tolist())
    hard_exclude = rated_ids | set(disliked_ids) | set(skipped_ids) \
        | set(interested_ids) | set(not_interested_ids)
    for i, mid in enumerate(movies['movieId'].tolist()):
        if mid in hard_exclude:
            hybrid[i] = -1.0

    score_map = {int(mid): float(hybrid[i])
                 for i, mid in enumerate(movies['movieId'].tolist())}
    candidates = [mid for mid, s in score_map.items() if s > 0]

    # ---- Diversity re-ranking (MMR) ----
    diversity = float(request.cookies.get(_pp('diversity'), 0.25) or 0.25)
    main_ids = rec.mmr_rerank(
        candidates, score_map, ITEM_MATRIX, movies,
        diversity=diversity, top_k=12,
    )

    # ---- "Because you liked ..." list (content-only) ----
    similar_ids = []
    if liked_ids or interested_ids:
        cb_only = cb_scores_vec.copy()
        for i, mid in enumerate(movies['movieId'].tolist()):
            if mid in hard_exclude or mid in liked_ids:
                cb_only[i] = -1.0
        cb_map = {int(mid): float(cb_only[i])
                  for i, mid in enumerate(movies['movieId'].tolist())}
        cb_cands = [m for m, s in cb_map.items() if s > 0]
        similar_ids = rec.mmr_rerank(
            cb_cands, cb_map, ITEM_MATRIX, movies,
            diversity=diversity * 0.5, top_k=12,
        )

    # ---- Explanations ----
    explain_ids = list(dict.fromkeys(main_ids + similar_ids))
    explanations = rec.build_explanations(
        explain_ids, movies, liked_ids, user_rates_df,
        cf_pred=cf_pred_map, cb_scores=cb_scores_vec,
        item_matrix=ITEM_MATRIX,
    )

    main_records = _movies_in_order(main_ids)
    similar_records = _movies_in_order(similar_ids)

    return (
        main_records,
        similar_records,
        "Hybrid of time-decayed collaborative filtering and content-based filtering.",
        "Based on the plot and genres of the movies you liked.",
        explanations,
    )


def _movies_in_order(movie_ids):
    mid_to_row = {mid: i for i, mid in enumerate(movies['movieId'].tolist())}
    rows = [mid_to_row[m] for m in movie_ids if m in mid_to_row]
    return movies.iloc[rows].to_dict('records') if rows else []


# ===========================================================================
# JSON APIs consumed by the enhanced front-end
# ===========================================================================
@bp.route('/api/search', methods=('GET',))
def api_search():
    q = (request.args.get('q') or '').strip().lower()
    if len(q) < 2:
        return jsonify([])
    mask = movies['title'].str.lower().str.contains(q, na=False)
    hits = movies[mask].head(20)[
        ['movieId', 'title', 'release_date', 'cover_url', 'genres']
    ].to_dict('records')
    return jsonify(hits)


@bp.route('/api/movie/<int:movie_id>', methods=('GET',))
def api_movie(movie_id):
    row = movies[movies['movieId'] == movie_id]
    if row.empty:
        return jsonify({}), 404
    m = row.iloc[0].to_dict()
    m['genres'] = list(m.get('genres') or [])
    return jsonify(m)


@bp.route('/api/explain/<int:movie_id>', methods=('GET',))
def api_explain(movie_id):
    liked_ids = [int(x) for x in _cookie_list('user_likes')]
    rates_raw = _cookie_list('user_rates')
    user_rates_df = ratesFromUser(rates_raw, keep_timestamp=True) \
        if rates_raw else None
    explanations = rec.build_explanations(
        [movie_id], movies, liked_ids, user_rates_df,
        item_matrix=ITEM_MATRIX,
    )
    return jsonify(explanations.get(movie_id, {'text': '', 'similar_to': [],
                                                'shared_genres': []}))


@bp.route('/api/profile/stats', methods=('GET',))
def api_profile_stats():
    """Numbers for the 'Your Taste' dashboard."""
    rates_raw = _cookie_list('user_rates')
    likes_raw = _cookie_list('user_likes')
    interests_raw = _cookie_list('user_interests')
    interested_ids, _not_int, _skip = _parse_interests(interests_raw)

    ratings_list = []
    for e in rates_raw:
        parts = e.split('|')
        if len(parts) >= 3:
            try:
                ratings_list.append((int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    rated_ids = [m for m, _ in ratings_list]
    liked_ids = [int(x) for x in likes_raw if x]

    # ---- Genre averages (across all rated/liked/interested movies) ----
    genre_totals = {}
    genre_counts = {}
    def _add(mid, score):
        row = movies[movies['movieId'] == mid]
        if row.empty:
            return
        for g in (row.iloc[0]['genres'] or []):
            genre_totals[g] = genre_totals.get(g, 0.0) + score
            genre_counts[g] = genre_counts.get(g, 0) + 1
    for mid, r in ratings_list:
        _add(mid, float(r))
    for mid in liked_ids:
        _add(mid, 5.0)
    for mid in interested_ids:
        _add(mid, 4.0)
    genre_ratings = {g: round(genre_totals[g] / genre_counts[g], 2)
                     for g in genre_totals}

    # ---- Top TF-IDF keywords from liked / highly-rated / interested ----
    mid_to_row = {mid: i for i, mid in enumerate(movies['movieId'].tolist())}
    seed_rows = []
    for mid in liked_ids:
        if mid in mid_to_row:
            seed_rows.append(mid_to_row[mid])
    for mid, r in ratings_list:
        if r >= 4 and mid in mid_to_row:
            seed_rows.append(mid_to_row[mid])
    for mid in interested_ids:
        if mid in mid_to_row:
            seed_rows.append(mid_to_row[mid])
    top_keywords = []
    if seed_rows:
        # The item matrix is [tfidf | genre_block]; restrict to the tfidf part.
        n_text = _TFIDF.get_feature_names_out().shape[0] if hasattr(
            _TFIDF, 'get_feature_names_out') else len(_TFIDF.vocabulary_)
        vec = np.asarray(ITEM_MATRIX[seed_rows, :n_text].sum(axis=0)).ravel()
        if vec.max() > 0:
            top_idx = np.argsort(vec)[::-1][:8]
            vocab = (_TFIDF.get_feature_names_out()
                     if hasattr(_TFIDF, 'get_feature_names_out')
                     else np.array(_TFIDF.get_feature_names()))
            top_keywords = [
                {'word': str(vocab[i]), 'weight': float(vec[i])}
                for i in top_idx if vec[i] > 0
            ]

    # ---- Decade distribution from rated/liked/interested years ----
    decade_counts = {}
    for mid in set(rated_ids + liked_ids + interested_ids):
        row = movies[movies['movieId'] == mid]
        if row.empty:
            continue
        y = row.iloc[0].get('year')
        if pd.isna(y):
            continue
        d = f'{int(y)//10*10}s'
        decade_counts[d] = decade_counts.get(d, 0) + 1

    return jsonify({
        'profile_id': _current_profile(),
        'genre_ratings': genre_ratings,
        'top_keywords': top_keywords,
        'decade_counts': decade_counts,
        'totals': {
            'rated': len(rated_ids),
            'liked': len(liked_ids),
            'interested': len(interested_ids),
        },
    })
