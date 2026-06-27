"""
Offline smoke test for the enhanced recommender pipeline.

Runs the full enhanced stack (time-decayed kNN + TF-IDF content model +
hybrid fusion + MMR + explanations) without starting Flask, so you can
sanity-check algorithm changes quickly.

    python scripts/smoke_test_recommenders.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flaskr.tools.data_tool import loadData, ratesFromUser
from flaskr.tools import recommenders as rec


def main() -> None:
    print("Loading data...")
    movies, genres, rates = loadData()
    print(f"  movies={len(movies)}  genres={len(genres)}  rates={len(rates)}")

    print("Building TF-IDF + genre item matrix...")
    t0 = time.time()
    item_matrix, _tfidf, genre_list = rec.build_content_matrix(movies)
    print(f"  matrix shape={item_matrix.shape}  (took {time.time()-t0:.1f}s)")

    now = int(time.time())
    fake_rates_raw = [
        f"611|1|5|{now - 30 * 86400}",      # Toy Story (1995)
        f"611|260|5|{now - 60 * 86400}",    # Star Wars
        f"611|318|5|{now - 20 * 86400}",    # Shawshank Redemption
        f"611|2959|4|{now - 100 * 86400}",  # Fight Club
        f"611|4993|4|{now - 70 * 86400}",   # LOTR: Fellowship
        f"611|5952|4|{now - 65 * 86400}",   # LOTR: Two Towers
        f"611|7153|4|{now - 60 * 86400}",   # LOTR: Return of the King
        f"611|356|4|{now - 200 * 86400}",   # Forrest Gump
        f"611|296|5|{now - 300 * 86400}",   # Pulp Fiction
        f"611|593|4|{now - 500 * 86400}",   # Silence of the Lambs
    ]
    user_rates_df = ratesFromUser(fake_rates_raw, keep_timestamp=True)
    liked_ids = [4993, 5952]

    print("Running time-decayed kNN...")
    t0 = time.time()
    cf_top, cf_pred, _ = rec.time_decayed_knn(
        rates[["userId", "movieId", "rating", "timestamp"]],
        user_rates_df, movies["movieId"].tolist(), user_id=611, top_k=12,
    )
    print(f"  top-5 CF ids: {cf_top[:5]}  (took {time.time()-t0:.1f}s)")

    print("Running content-based scoring...")
    rated_map = {int(r.movieId): float(r.rating) / 5.0
                 for r in user_rates_df.itertuples() if r.rating >= 3}
    cb_scores = rec.content_based_scores(
        movies, item_matrix, liked_ids, rated_ids_with_weight=rated_map,
    )
    print(f"  cb shape={cb_scores.shape}  max={cb_scores.max():.3f}")

    cf_scores_vec = np.zeros(len(movies))
    for i, mid in enumerate(movies["movieId"].tolist()):
        if mid in cf_pred:
            cf_scores_vec[i] = cf_pred[mid]

    hybrid = rec.hybrid_scores(cf_scores_vec, cb_scores, alpha=0.6)
    rated_ids = set(user_rates_df["movieId"].tolist())
    for i, mid in enumerate(movies["movieId"].tolist()):
        if mid in rated_ids:
            hybrid[i] = -1.0
    score_map = {int(mid): float(hybrid[i])
                 for i, mid in enumerate(movies["movieId"].tolist())}
    cand = [m for m, s in score_map.items() if s > 0]

    main_ids = rec.mmr_rerank(cand, score_map, item_matrix, movies,
                              diversity=0.3, top_k=12)
    print("Top-12 hybrid+MMR recommendations:")
    for mid in main_ids:
        title = movies.loc[movies["movieId"] == mid, "title"].iloc[0]
        print(f"  {mid:>6}  {title}")

    explanations = rec.build_explanations(
        main_ids, movies, liked_ids, user_rates_df,
        cf_pred=cf_pred, cb_scores=cb_scores, item_matrix=item_matrix,
    )
    print("\nSample explanations:")
    for mid in main_ids[:3]:
        print(f"  [{mid}] {explanations[mid]['text']}")

    print("\nSmoke test OK.")


if __name__ == "__main__":
    main()
