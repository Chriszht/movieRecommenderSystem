"""
Enhanced recommendation algorithms for the COMP4135 project.

Implements three course-covered methods that leverage signals ignored by
the original demo (timestamp, movie overview, text content):

    A1.  Time-decayed user-based kNN (collaborative filtering).
    A2.  TF-IDF + genre multi-hot content-based filtering.
    A3.  Hybrid linear-combination score + MMR diversity re-ranking.

Each public function also returns per-item explanation fragments so the
front-end can show "why this was recommended".
"""

import math
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, KNNWithMeans


# --------------------------------------------------------------------------
# A1. Time-decayed user-based kNN
# --------------------------------------------------------------------------
def time_decayed_knn(all_rates, user_rates_df, all_movie_ids, user_id=611,
                     half_life_days=365, k_neighbors=40, top_k=12):
    """
    Exponentially decay each historical rating by its age so that recent
    tastes weigh more than decade-old ratings. The decayed rating is used
    as the training signal for user-based kNN-with-means.

    Returns
    -------
    (top_movie_ids, predicted_scores, neighbor_info)
        top_movie_ids    : list[int]                  top-K predicted movieIds
        predicted_scores : dict[int, float]           movieId -> predicted rating
        neighbor_info    : dict[int, list[(uid, sim)]] movieId -> supporting neighbors
    """
    training_rates = pd.concat([all_rates, user_rates_df], ignore_index=True)
    training_rates = training_rates.dropna(subset=['timestamp'])

    latest_ts = training_rates['timestamp'].max()
    half_life_seconds = half_life_days * 24 * 3600
    decay = np.exp(-(latest_ts - training_rates['timestamp']) *
                   math.log(2) / half_life_seconds)

    # Shift decayed ratings back to the 1-5 range: mean-centre then re-scale.
    mean_r = training_rates['rating'].mean()
    decayed = mean_r + (training_rates['rating'] - mean_r) * decay
    training_rates = training_rates.assign(rating=decayed.clip(1, 5))

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        training_rates[['userId', 'movieId', 'rating']], reader=reader
    )
    trainset = data.build_full_trainset()
    algo = KNNWithMeans(
        k=k_neighbors,
        sim_options={'name': 'pearson', 'user_based': True},
        verbose=False,
    )
    algo.fit(trainset)

    rated_ids = set(user_rates_df['movieId'].tolist())
    predictions, neighbor_info = [], {}
    try:
        inner_uid = trainset.to_inner_uid(user_id)
    except ValueError:
        inner_uid = None

    for mid in all_movie_ids:
        if mid in rated_ids:
            continue
        p = algo.predict(user_id, mid)
        predictions.append((mid, p.est))
        # Capture a small explanation trail: top-3 most-similar neighbours
        # among users that actually rated this item.
        if inner_uid is not None:
            try:
                inner_iid = trainset.to_inner_iid(mid)
                rater_uids = [u for (u, _) in trainset.ir[inner_iid]]
                sims = sorted(
                    ((ru, float(algo.sim[inner_uid, ru])) for ru in rater_uids),
                    key=lambda x: x[1], reverse=True,
                )[:3]
                neighbor_info[mid] = [
                    (int(trainset.to_raw_uid(ru)), s) for ru, s in sims
                ]
            except ValueError:
                neighbor_info[mid] = []
        else:
            neighbor_info[mid] = []

    predictions.sort(key=lambda x: x[1], reverse=True)
    top = predictions[:top_k]
    return (
        [mid for mid, _ in top],
        {mid: score for mid, score in predictions},
        neighbor_info,
    )


# --------------------------------------------------------------------------
# A2. TF-IDF + genre content-based filtering
# --------------------------------------------------------------------------
def build_content_matrix(movies_df, genre_weight=2.0):
    """
    Build a sparse TF-IDF item matrix from the movie *overview* text,
    concatenated with a weighted multi-hot *genre* block. Returned once and
    cached by the caller (loaded at app start).
    """
    overviews = movies_df['overview'].fillna('').astype(str).tolist()
    tfidf = TfidfVectorizer(
        stop_words='english', max_features=5000,
        ngram_range=(1, 2), min_df=2,
    )
    text_mat = tfidf.fit_transform(overviews)

    # Multi-hot genre encoding.
    genres_series = movies_df['genres'].apply(
        lambda g: g if isinstance(g, list) else []
    )
    genre_list = sorted({g for gs in genres_series for g in gs})
    genre_idx = {g: i for i, g in enumerate(genre_list)}
    rows, cols = [], []
    for i, gs in enumerate(genres_series):
        for g in gs:
            rows.append(i)
            cols.append(genre_idx[g])
    data = np.ones(len(rows)) * genre_weight
    genre_mat = csr_matrix(
        (data, (rows, cols)),
        shape=(len(movies_df), len(genre_list)),
    )

    item_matrix = hstack([text_mat, genre_mat]).tocsr()
    return item_matrix, tfidf, genre_list


def content_based_scores(movies_df, item_matrix, liked_ids,
                         rated_ids_with_weight=None, interested_ids=None):
    """
    Build a user profile as a *weighted* average of the item vectors the
    user has liked (and optionally highly-rated items), then score every
    movie by cosine similarity.

    Parameters
    ----------
    rated_ids_with_weight : dict[int, float] | None
        Optional extra signal: {movieId: rating/5.0}. Ratings are given
        half the weight of explicit likes, mirroring Koren et al.'s
        implicit-feedback weighting.
    interested_ids : list[int] | None
        Movies the user has not seen but declared interest in; half the
        weight of an explicit like (they reflect taste, not experience).

    Returns
    -------
    scores : np.ndarray shape=(n_movies,)
    """
    movie_id_to_row = {mid: i for i, mid in enumerate(movies_df['movieId'].tolist())}

    rows, weights = [], []
    for mid in liked_ids:
        if mid in movie_id_to_row:
            rows.append(movie_id_to_row[mid])
            weights.append(1.0)
    if rated_ids_with_weight:
        for mid, w in rated_ids_with_weight.items():
            if mid in movie_id_to_row:
                rows.append(movie_id_to_row[mid])
                weights.append(0.5 * float(w))
    if interested_ids:
        for mid in interested_ids:
            if mid in movie_id_to_row:
                rows.append(movie_id_to_row[mid])
                weights.append(0.5)

    if not rows:
        return np.zeros(item_matrix.shape[0])

    weights = np.asarray(weights).reshape(-1, 1)
    profile = item_matrix[rows].multiply(weights).sum(axis=0)
    profile = np.asarray(profile)
    # Normalise to a unit vector.
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm
    scores = cosine_similarity(profile, item_matrix)[0]
    return scores


# --------------------------------------------------------------------------
# A3. Hybrid + MMR diversity re-ranking
# --------------------------------------------------------------------------
def hybrid_scores(cf_scores, cb_scores, alpha=0.6):
    """
    Min-max normalise each score vector to [0,1] then take a convex
    combination.  alpha weights the collaborative-filtering signal.
    """
    def _norm(v):
        v = np.asarray(v, dtype=float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        if hi - lo < 1e-9:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    return alpha * _norm(cf_scores) + (1.0 - alpha) * _norm(cb_scores)


def mmr_rerank(candidate_ids, score_map, item_matrix, movies_df,
               diversity=0.3, top_k=12):
    """
    Maximal Marginal Relevance: greedy selection that trades off
    relevance (score_map) against novelty (1 - max_similarity_to_selected).
    diversity=0 keeps the pure ranking; diversity=1 picks the most
    dissimilar items.
    """
    if diversity <= 0 or len(candidate_ids) <= top_k:
        ranked = sorted(candidate_ids, key=lambda m: score_map.get(m, 0),
                        reverse=True)
        return ranked[:top_k]

    movie_id_to_row = {mid: i for i, mid in enumerate(movies_df['movieId'].tolist())}
    cand = sorted(candidate_ids, key=lambda m: score_map.get(m, 0),
                  reverse=True)[:max(top_k * 4, 40)]
    cand_rows = [movie_id_to_row[m] for m in cand if m in movie_id_to_row]
    cand = [m for m in cand if m in movie_id_to_row]
    sub = item_matrix[cand_rows]
    sim = cosine_similarity(sub)

    selected, selected_idx = [], []
    remaining = list(range(len(cand)))
    while remaining and len(selected) < top_k:
        best_i, best_v = None, -1e9
        for i in remaining:
            rel = score_map.get(cand[i], 0.0)
            div_pen = max((sim[i, j] for j in selected_idx), default=0.0)
            v = (1 - diversity) * rel - diversity * div_pen
            if v > best_v:
                best_v, best_i = v, i
        selected.append(cand[best_i])
        selected_idx.append(best_i)
        remaining.remove(best_i)
    return selected


# --------------------------------------------------------------------------
# Explanation helper
# --------------------------------------------------------------------------
def build_explanations(rec_ids, movies_df, user_likes, user_rates_df,
                       cf_pred=None, cb_scores=None, item_matrix=None,
                       top_reason_items=2):
    """
    Produce a short human-readable reason for each recommended movie.

    Returns dict[int, dict]: {movieId: {text, shared_genres, similar_to}}.
    """
    out = {}
    mid_to_row = {mid: i for i, mid in enumerate(movies_df['movieId'].tolist())}
    reference_ids = list(user_likes)
    if user_rates_df is not None and len(user_rates_df) > 0:
        high_rated = user_rates_df[user_rates_df['rating'] >= 4]['movieId'].tolist()
        reference_ids.extend(m for m in high_rated if m not in reference_ids)

    valid_refs = [m for m in reference_ids if m in mid_to_row]
    ref_rows = [mid_to_row[m] for m in valid_refs]
    ref_sim_matrix = None
    if item_matrix is not None and ref_rows:
        ref_sim_matrix = cosine_similarity(item_matrix[ref_rows], item_matrix)

    for mid in rec_ids:
        if mid not in mid_to_row:
            continue
        row = mid_to_row[mid]
        rec_genres = set(movies_df.iloc[row]['genres'] or [])
        # Most similar previously-rated/liked movies.
        similar_to = []
        if ref_sim_matrix is not None:
            sims = ref_sim_matrix[:, row]
            order = np.argsort(sims)[::-1][:top_reason_items]
            for k in order:
                ref_mid = valid_refs[k]
                if sims[k] <= 0 or ref_mid == mid:
                    continue
                similar_to.append({
                    'movieId': int(ref_mid),
                    'title': str(movies_df.iloc[mid_to_row[ref_mid]]['title']),
                    'similarity': float(sims[k]),
                })
        shared = []
        for ref_mid in valid_refs[:20]:
            r_genres = set(movies_df.iloc[mid_to_row[ref_mid]]['genres'] or [])
            shared.extend(rec_genres & r_genres)
        shared_genres = sorted(set(shared))[:3]
        parts = []
        if similar_to:
            parts.append("Similar to " + ", ".join(s['title'] for s in similar_to))
        if shared_genres:
            parts.append("Shared genres: " + ", ".join(shared_genres))
        if cf_pred is not None and mid in cf_pred:
            parts.append(f"Predicted rating {cf_pred[mid]:.1f}/5")
        out[int(mid)] = {
            'text': " \u00b7 ".join(parts) if parts else "Popular with users like you.",
            'shared_genres': shared_genres,
            'similar_to': similar_to,
        }
    return out
