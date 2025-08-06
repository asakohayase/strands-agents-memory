from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from enum import Enum


class Genre(str, Enum):
    ACTION = "action"
    COMEDY = "comedy"
    DRAMA = "drama"
    HORROR = "horror"
    ROMANCE = "romance"
    SCI_FI = "sci-fi"
    THRILLER = "thriller"
    FANTASY = "fantasy"
    DOCUMENTARY = "documentary"
    ANIMATION = "animation"


@dataclass
class Movie:
    id: str
    title: str
    year: int
    genres: List[Genre]
    rating: float
    series: str = None  # For grouping series movies

    def to_dict(self) -> Dict:
        return asdict(self)


# Movie database with series information
MOVIE_DATABASE = {
    # Matrix Series
    "1": Movie("1", "The Matrix", 1999, [Genre.SCI_FI, Genre.ACTION], 8.7, "matrix"),
    "2": Movie(
        "2", "Matrix Reloaded", 2003, [Genre.SCI_FI, Genre.ACTION], 7.2, "matrix"
    ),
    "3": Movie(
        "3", "Matrix Revolutions", 2003, [Genre.SCI_FI, Genre.ACTION], 6.8, "matrix"
    ),
    "4": Movie(
        "4", "Matrix Resurrections", 2021, [Genre.SCI_FI, Genre.ACTION], 5.7, "matrix"
    ),
    # Star Wars Series
    "5": Movie(
        "5",
        "Star Wars: A New Hope",
        1977,
        [Genre.SCI_FI, Genre.ACTION],
        8.6,
        "star_wars",
    ),
    "6": Movie(
        "6",
        "Star Wars: The Empire Strikes Back",
        1980,
        [Genre.SCI_FI, Genre.ACTION],
        8.7,
        "star_wars",
    ),
    "7": Movie(
        "7",
        "Star Wars: Return of the Jedi",
        1983,
        [Genre.SCI_FI, Genre.ACTION],
        8.3,
        "star_wars",
    ),
    # Classic Movies
    "8": Movie("8", "Inception", 2010, [Genre.SCI_FI, Genre.THRILLER], 8.8),
    "9": Movie("9", "The Grand Budapest Hotel", 2014, [Genre.COMEDY, Genre.DRAMA], 8.1),
    "10": Movie("10", "Parasite", 2019, [Genre.THRILLER, Genre.DRAMA], 8.6),
    "11": Movie("11", "Spirited Away", 2001, [Genre.ANIMATION, Genre.FANTASY], 9.3),
    "12": Movie("12", "Get Out", 2017, [Genre.HORROR, Genre.THRILLER], 7.7),
    "13": Movie("13", "La La Land", 2016, [Genre.ROMANCE, Genre.DRAMA], 8.0),
    "14": Movie("14", "Mad Max: Fury Road", 2015, [Genre.ACTION, Genre.THRILLER], 8.1),
    "15": Movie("15", "Her", 2013, [Genre.ROMANCE, Genre.SCI_FI], 8.0),
    # 2023 Top Movies
    "16": Movie("16", "Oppenheimer", 2023, [Genre.DRAMA, Genre.THRILLER], 8.3),
    "17": Movie("17", "Barbie", 2023, [Genre.COMEDY, Genre.FANTASY], 6.9),
    "18": Movie(
        "18", "Guardians of the Galaxy Vol. 3", 2023, [Genre.ACTION, Genre.SCI_FI], 7.9
    ),
    "19": Movie(
        "19",
        "Spider-Man: Across the Spider-Verse",
        2023,
        [Genre.ANIMATION, Genre.ACTION],
        8.7,
    ),
    "20": Movie(
        "20", "John Wick: Chapter 4", 2023, [Genre.ACTION, Genre.THRILLER], 7.7
    ),
    "21": Movie("21", "Scream VI", 2023, [Genre.HORROR, Genre.THRILLER], 6.5),
    "22": Movie("22", "The Little Mermaid", 2023, [Genre.FANTASY, Genre.ROMANCE], 7.2),
    "23": Movie("23", "Fast X", 2023, [Genre.ACTION, Genre.THRILLER], 5.8),
    "24": Movie(
        "24", "The Super Mario Bros. Movie", 2023, [Genre.ANIMATION, Genre.COMEDY], 7.0
    ),
    "25": Movie("25", "Cocaine Bear", 2023, [Genre.COMEDY, Genre.THRILLER], 5.9),
    # 2024 Top Movies
    "26": Movie("26", "Dune: Part Two", 2024, [Genre.SCI_FI, Genre.ACTION], 8.5),
    "27": Movie("27", "Inside Out 2", 2024, [Genre.ANIMATION, Genre.COMEDY], 7.6),
    "28": Movie("28", "Deadpool & Wolverine", 2024, [Genre.ACTION, Genre.COMEDY], 7.7),
    "29": Movie("29", "Wicked: Part One", 2024, [Genre.FANTASY, Genre.ROMANCE], 8.2),
    "30": Movie(
        "30", "A Quiet Place: Day One", 2024, [Genre.HORROR, Genre.THRILLER], 6.7
    ),
    "31": Movie("31", "Bad Boys: Ride or Die", 2024, [Genre.ACTION, Genre.COMEDY], 6.6),
    "32": Movie(
        "32", "Beetlejuice Beetlejuice", 2024, [Genre.COMEDY, Genre.HORROR], 7.0
    ),
    "33": Movie("33", "Terrifier 3", 2024, [Genre.HORROR], 6.9),
    "34": Movie("34", "Gladiator II", 2024, [Genre.ACTION, Genre.DRAMA], 6.8),
    "35": Movie("35", "Moana 2", 2024, [Genre.ANIMATION, Genre.FANTASY], 7.0),
    # 2025 Anticipated/Early Releases
    "36": Movie(
        "36",
        "Captain America: Brave New World",
        2025,
        [Genre.ACTION, Genre.SCI_FI],
        7.5,
    ),
    "37": Movie("37", "Thunderbolts", 2025, [Genre.ACTION, Genre.COMEDY], 7.2),
    "38": Movie(
        "38", "The Fantastic Four: First Steps", 2025, [Genre.ACTION, Genre.SCI_FI], 7.3
    ),
    "39": Movie("39", "Superman", 2025, [Genre.ACTION, Genre.SCI_FI], 7.8),
    "40": Movie(
        "40", "How to Train Your Dragon", 2025, [Genre.ANIMATION, Genre.FANTASY], 7.4
    ),
    "41": Movie("41", "Lilo & Stitch", 2025, [Genre.ANIMATION, Genre.COMEDY], 7.1),
    "42": Movie("42", "The Batman Part II", 2025, [Genre.ACTION, Genre.THRILLER], 8.0),
    "43": Movie("43", "Blade", 2025, [Genre.ACTION, Genre.HORROR], 7.0),
    "44": Movie("44", "Avatar: Fire and Ash", 2025, [Genre.SCI_FI, Genre.ACTION], 7.9),
    "45": Movie(
        "45", "Mission: Impossible 8", 2025, [Genre.ACTION, Genre.THRILLER], 7.6
    ),
    # Additional Popular Movies to Round Out Genres
    "46": Movie("46", "The Notebook", 2004, [Genre.ROMANCE, Genre.DRAMA], 7.8),
    "47": Movie("47", "Titanic", 1997, [Genre.ROMANCE, Genre.DRAMA], 7.9),
    "48": Movie("48", "The Conjuring", 2013, [Genre.HORROR, Genre.THRILLER], 7.5),
    "49": Movie("49", "Hereditary", 2018, [Genre.HORROR, Genre.THRILLER], 7.3),
    "50": Movie("50", "Won't You Be My Neighbor?", 2018, [Genre.DOCUMENTARY], 8.4),
}


def get_movie_by_title(title: str) -> Movie:
    """Find movie by title (case insensitive partial match)"""
    title_lower = title.lower()
    for movie in MOVIE_DATABASE.values():
        if title_lower in movie.title.lower():
            return movie
    return None


def get_movies_by_series(series: str) -> List[Movie]:
    """Get all movies in a series"""
    return [movie for movie in MOVIE_DATABASE.values() if movie.series == series]


def get_all_movies() -> List[Movie]:
    """Get all movies in database"""
    return list(MOVIE_DATABASE.values())
