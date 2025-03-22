import requests
from geopy.geocoders import Nominatim

def get_location_from_ip():
    # Use IP-based geolocation to obtain the approximate location
    response = requests.get('https://ipinfo.io')
    data = response.json()
    loc = data['loc']
    latitude, longitude = loc.split(',')
    return float(latitude), float(longitude)

def get_location(latitude, longitude):
    geolocator = Nominatim(user_agent="location_finder")
    location = geolocator.reverse((latitude, longitude))
    return location.address

def main():
    latitude, longitude = get_location_from_ip()
    location = get_location(latitude, longitude)
    print("Location:", location)
    print("Latitude:", latitude)
    print("Longitude:", longitude)

if __name__ == "__main__":
    main()
