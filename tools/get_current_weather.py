import requests

def get_current_weather(location, unit='celsius'):
    api_key = None  # Replace with a real API key
    base_url = 'http://api.openweathermap.org/data/2.5/weather'

    # Fall back to use DuckDuckGo search if the API key is missing
    if api_key is None:
        try:
            from duckduckgo_search import DDGS
            results = DDGS().text(f"weather in {location} in {unit}")
            return results
        except Exception as e:
            return f"Error: {str(e)}"

    params = {
        'q': location,
        'appid': api_key,
        'units': 'imperial' if unit == 'fahrenheit' else 'metric'
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        temperature = data['main']['temp']
        description = data['weather'][0]['description']

        return f'The current weather in {location} is {temperature}°{unit[0].upper()} with {description}.'
    else:
        return f'Error: Unable to fetch weather data for {location}.'