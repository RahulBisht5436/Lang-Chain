import requests


def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.

    The function performs two API calls:
    1. Geocoding API → converts city name into latitude/longitude.
    2. Weather API → uses latitude/longitude to get current weather.
    """

    # ---------------------------------------------------------
    # STEP 1: Convert the city name into latitude & longitude
    # ---------------------------------------------------------
    # Open-Meteo provides a Geocoding API that can search for
    # a city and return its geographical coordinates.
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"

    # Send a GET request to the Geocoding API.
    #
    # "params" contains query parameters that will be added
    # automatically to the URL.
    #
    # Example:
    # ?name=Noida&count=1&language=en&format=json
    geo_response = requests.get(
        geo_url,
        params={
            "name": city,       # City name entered by the user
            "count": 1,         # Return only the first matching result
            "language": "en",   # Return results in English
            "format": "json"    # Ask the API to return JSON
        }
    )

    # Convert the HTTP response from JSON format
    # into a Python dictionary.
    geo_data = geo_response.json()

    # Check whether the API returned any search results.
    #
    # If "results" doesn't exist, it means the city could not
    # be found by the Geocoding API.
    if "results" not in geo_data:
        return f"Could not find the city: {city}"

    # Get the first city returned by the API.
    #
    # Because we used count=1, we expect only one result.
    location = geo_data["results"][0]

    # Extract latitude and longitude from the result.
    #
    # These coordinates will be used by the Weather API
    # to determine the exact location for which we want weather.
    latitude = location["latitude"]
    longitude = location["longitude"]

    # ---------------------------------------------------------
    # STEP 2: Get current weather using the coordinates
    # ---------------------------------------------------------
    # This is the Open-Meteo Forecast API endpoint.
    weather_url = "https://api.open-meteo.com/v1/forecast"

    # Send another GET request, this time to the Weather API.
    weather_response = requests.get(
        weather_url,
        params={
            # Tell the API which geographical location
            # we want weather information for.
            "latitude": latitude,
            "longitude": longitude,

            # Ask the API to return these current weather values:
            #
            # temperature_2m       → temperature at 2 meters
            # relative_humidity_2m → relative humidity at 2 meters
            # weather_code         → WMO weather condition code
            # wind_speed_10m       → wind speed at 10 meters
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "weather_code,"
                "wind_speed_10m"
            ),

            # Automatically use the timezone of the requested location.
            "timezone": "auto"
        }
    )

    # Convert the weather API response from JSON
    # into a Python dictionary.
    weather_data = weather_response.json()

    # Extract the "current" section from the API response.
    #
    # This contains the current temperature, humidity,
    # wind speed, weather code, etc.
    current = weather_data["current"]

    # ---------------------------------------------------------
    # STEP 3: Create a human-readable response
    # ---------------------------------------------------------
    # Extract the required values from the "current" dictionary
    # and combine them into a single string.
    return (
        f"Weather in {city}: "
        f"Temperature: {current['temperature_2m']}°C, "
        f"Humidity: {current['relative_humidity_2m']}%, "
        f"Wind speed: {current['wind_speed_10m']} km/h, "
        f"Weather code: {current['weather_code']}"
    )