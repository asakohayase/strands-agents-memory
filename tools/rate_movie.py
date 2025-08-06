from strands import tool
from typing import Dict, Any, List
import sys
import os

# Add parent directory to path to import movie_database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from movie_database import MOVIE_DATABASE, get_movie_by_title, get_movies_by_series


@tool
def rate_movie(movie_title: str, user_rating: float, liked: bool) -> Dict[str, Any]:
    """
    Rate a movie and automatically apply to series if applicable.

    Args:
        movie_title: Name of the movie (can be partial, like "Matrix")
        user_rating: Rating from 1-5 stars
        liked: Whether user liked the movie

    Returns:
        Dict with rating information and memory content to store
    """
    # Find matching movie
    matched_movie = get_movie_by_title(movie_title)

    if not matched_movie:
        return {
            "error": f"Movie '{movie_title}' not found",
            "suggestions": [m.title for m in list(MOVIE_DATABASE.values())[:5]],
        }

    # Get all movies in the series (if applicable)
    movies_to_rate = [matched_movie]
    if matched_movie.series:
        movies_to_rate = get_movies_by_series(matched_movie.series)

    # Create memory content for each movie
    memory_entries = []
    genre_info = [g.value for g in matched_movie.genres]

    for movie in movies_to_rate:
        memory_content = f"User rated '{movie.title}' {user_rating}/5 stars. "
        memory_content += f"User {'liked' if liked else 'disliked'} this movie. "
        memory_content += f"Genres: {', '.join([g.value for g in movie.genres])}."
        memory_entries.append(memory_content)

    # Create genre preference memory
    genre_preference = f"User {'likes' if liked and user_rating >= 4 else 'dislikes'} {', '.join(genre_info)} movies."
    memory_entries.append(genre_preference)

    return {
        "success": True,
        "rated_movies": [m.title for m in movies_to_rate],
        "memory_entries": memory_entries,
        "message": f"Rated {len(movies_to_rate)} movie(s) with {user_rating}/5 stars",
        "applied_to_series": len(movies_to_rate) > 1,
    }
