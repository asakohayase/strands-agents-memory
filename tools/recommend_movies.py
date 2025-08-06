from strands import tool
from typing import Dict, Any
import sys
import os

# Add parent directory to path to import movie_database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from movie_database import MOVIE_DATABASE, get_all_movies


@tool
def recommend_movies(
    user_memories: str = "", count: int = 5, genre_filter: str = None
) -> Dict[str, Any]:
    """
    Generate movie recommendations based on user's stored preferences.

    Args:
        user_memories: String containing user's movie ratings and preferences from memory
        count: Number of recommendations to return
        genre_filter: Optional genre to filter by (e.g., "comedy", "sci-fi")

    Returns:
        Dict with personalized movie recommendations
    """
    # DEBUG: Log all parameters received
    print(f"🔍 DEBUG recommend_movies called with:")
    print(f"   user_memories: '{user_memories}'")
    print(f"   count: {count}")
    print(f"   genre_filter: '{genre_filter}'")

    all_movies = get_all_movies()

    # Filter by genre if specified
    if genre_filter:
        genre_filter_lower = genre_filter.lower()
        filtered_movies = [
            movie
            for movie in all_movies
            if any(genre_filter_lower in g.value.lower() for g in movie.genres)
        ]
        print(f"🔍 DEBUG: Filtered to {len(filtered_movies)} {genre_filter} movies")
        print(
            f"🔍 DEBUG: Comedy movies found: {[m.title for m in all_movies if 'comedy' in [g.value for g in m.genres]]}"
        )
    else:
        filtered_movies = all_movies
        print(f"🔍 DEBUG: No genre filter, using all {len(all_movies)} movies")

    if not user_memories:
        # No preferences stored, return popular movies (filtered by genre if specified)
        popular_movies = sorted(filtered_movies, key=lambda m: m.rating, reverse=True)[
            :count
        ]
        genre_msg = f" {genre_filter} movies" if genre_filter else " movies"

        result = {
            "recommendations": [
                {
                    "title": m.title,
                    "year": m.year,
                    "genres": [g.value for g in m.genres],
                    "rating": m.rating,
                    "reason": f"Highly rated{genre_msg}",
                }
                for m in popular_movies
            ],
            "note": f"No preferences found. Showing popular{genre_msg}.",
            "genre_filter_applied": genre_filter,
            "debug_info": {
                "total_movies": len(all_movies),
                "filtered_movies": len(filtered_movies),
                "genre_filter": genre_filter,
            },
        }

        print(f"🔍 DEBUG: Returning {len(result['recommendations'])} recommendations")
        return result

    # Parse memories to extract preferences
    memories_lower = user_memories.lower()
    liked_genres = set()
    disliked_genres = set()
    rated_movies = set()

    # Extract genre preferences from memories
    genres = [
        "sci-fi",
        "action",
        "comedy",
        "drama",
        "thriller",
        "horror",
        "romance",
        "fantasy",
        "animation",
    ]

    for genre in genres:
        if f"likes {genre}" in memories_lower:
            liked_genres.add(genre)
        elif f"dislikes {genre}" in memories_lower:
            disliked_genres.add(genre)

    # Extract rated movies to avoid recommending them
    for movie in all_movies:
        if movie.title.lower() in memories_lower:
            rated_movies.add(movie.title)

    # Score movies based on preferences (use filtered list)
    recommendations = []
    for movie in filtered_movies:
        if movie.title in rated_movies:
            continue  # Skip already rated movies

        score = movie.rating / 10.0  # Base score
        reason_parts = []

        # Genre scoring
        movie_genres = set([g.value for g in movie.genres])

        # Boost for liked genres
        liked_overlap = movie_genres & liked_genres
        if liked_overlap:
            score += 0.5
            reason_parts.append(f"You like {', '.join(liked_overlap)}")

        # Penalty for disliked genres
        disliked_overlap = movie_genres & disliked_genres
        if disliked_overlap:
            score -= 0.7
            continue  # Skip movies with disliked genres

        # Default reason if no specific preference match
        if not reason_parts:
            if genre_filter:
                reason_parts.append(f"Highly rated {genre_filter} movie")
            else:
                reason_parts.append("Highly rated movie")

        recommendations.append(
            {
                "title": movie.title,
                "year": movie.year,
                "genres": [g.value for g in movie.genres],
                "rating": movie.rating,
                "score": score,
                "reason": " and ".join(reason_parts),
            }
        )

    # Sort by score and return top recommendations
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    top_recs = recommendations[:count]

    return {
        "recommendations": [
            {
                "title": rec["title"],
                "year": rec["year"],
                "genres": rec["genres"],
                "rating": rec["rating"],
                "reason": rec["reason"],
            }
            for rec in top_recs
        ],
        "personalization_factors": {
            "liked_genres": list(liked_genres),
            "disliked_genres": list(disliked_genres),
            "rated_movies_count": len(rated_movies),
            "genre_filter": genre_filter,
        },
    }
