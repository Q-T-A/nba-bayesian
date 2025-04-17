BASE_URL = "https://www.basketball-reference.com/boxscores/"


def full_url(date):
    return f"{BASE_URL}/?month={date.month}&day={date.day}&year={date.year}"
