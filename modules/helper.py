#----------------------------------------------------
# Imports
#----------------------------------------------------
from modules import imports

#----------------------------------------------------
# Helper Functions
#----------------------------------------------------
def extract_sentiment_score(text: str) -> float:
    """
    Computes the sentiment score from a given text.

    text -> str
        Given input text

    Returns -> float
        Returns the sentiment score

    Example
        extract_sentiment_score("I love this!")
    """
    analyzer = imports.SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)["compound"]

def extract_duration_days(start: str, end: str) -> int:
    """
    Calculates the number of days that has occured between the start and end

    start -> str
        Given starting date

    end -> str
        Given ending date

    Returns -> int
        Returns the number of days that has elapsed

    Example
        extract_duration_days("1/1/2023", "1/2/2023")
    """
    start = imports.pd.to_datetime(start)
    end = imports.pd.to_datetime(end)
    duration = end - start
    return int(duration.days)

def calc_num_bins(num_scores: int) -> int:
    """
    Calculates the optimal number of histogram bins using Sturge's Rule

    num_scores -> int
        Given total number of datapoints that are considered in the histogram

    Returns -> int
        Returns the number of bins that optimally organizes the histogram

    Example
        calc_num_bins(500)
    """
    return int(1 + 3.322 * imports.math.log(num_scores))

def calc_limit_lwrbound(num_scores: list) -> float:
    """
    Calculates the optimal lowerbound for the y_lim when zooming in on values

    num_scores -> list
        Given iterable of values

    Returns -> float
        Returns the optimal lowerbound

    Example
        calc_limit_lwrbound([2,3,4,5])
    """
    min_value = min(num_scores)
    max_value = max(num_scores)
    range_value = max_value - min_value
    return min_value - range_value * 0.2

def get_season(date: str, mode: str="Northern") -> str:
    """
    Calculates the season of a month/day

    date -> str
        Given date to convert into a seaon

    mode -> str
        Determines whether to use the Northern or Southern Hemisphere calculation

    Return -> str
        Returns the season
        
    Example
        get_season("12/1/23")
    """
    # Check for correct mode
    options = ["Northern", "Southern"]
    if mode not in options:
        raise ValueError(f"{mode} invalid; must be within {options}.")

    # Convert the date to a datetime object
    date = imports.pd.to_datetime(date)
    
    # Source: https://en.wikipedia.org/wiki/Season#:~:text=According%20to%20this%20definition%2C%20for,and%20winter%20on%201%20June.
    if mode == "Northern":
        if (date.month >= 3 and date.month >= 1) and (date.month <= 5 and date.day <= 31):
            season = "Spring"
        elif (date.month >= 6 and date.month >= 1) and (date.month <= 8 and date.day <= 31):
            season = "Summer"
        elif (date.month >= 9 and date.month >= 1) and (date.month <= 11 and date.day <= 30):
            season = "Fall"
        else:
            season = "Winter"
    else: # Southern
        if (date.month >= 3 and date.month >= 1) and (date.month <= 5 and date.day <= 31):
            season = "Fall"
        elif (date.month >= 6 and date.month >= 1) and (date.month <= 8 and date.day <= 31):
            season = "Winter"
        elif (date.month >= 9 and date.month >= 1) and (date.month <= 11 and date.day <= 30):
            season = "Spring"
        else:
            season = "Summer"
    return season

#----------------------------------------------------
# Main
#----------------------------------------------------
def main():
    return None

#----------------------------------------------------
# Entry
#----------------------------------------------------
if __name__ == "__main__":
    main()