"""
Configuration for ESPN Fantasy Football Dashboard
Add your league IDs and year here
"""

LEAGUES = [
    {"league_id": 178016325, "year": 2026, "name": "Disgusting Myrtle PT. 2"},
    {"league_id": 1909924039, "year": 2026, "name": "Gambling Chat"},
]

CURRENT_YEAR = 2026

# ESPN Authentication Cookies (from browser developer tools)
ESPN_S2 = "AEBwTpmzhW8jxo2VQrvUh9Xz/9UfQN5ycpZdC8RV/ED18EQya1VkW5MCNo0XBY3rhlLHL/mcqwVg9gfrkz3h66Vu24Nqz5Jl+6TsXeRKtFSbjS1FzkTADmZQC+bK09yR7rWW10EwAbXxKC5TzqHI+1l+PRqsnl5JQz9dD3LP+kikNX2kU+ExlvLPMIMwodYmtjkmbTlyIvmkokjvsuaZX5CTB+nXlY4xJG19WXFKk67G192SqQpcvfFpcm2GCY+KXh1LD5K2VfajcYlrdaBF1NLyVn/NNfLQkZTYx/vFNz7lEA=="
SWID = "{D44D1A54-E98F-4200-9B0B-41C37A3C6510}"

# ESPN API settings
ESPN_API_YEAR_RANGE = range(CURRENT_YEAR - 5, CURRENT_YEAR + 1)
