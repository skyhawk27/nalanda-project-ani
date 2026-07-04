import json
# pyrefly: ignore [missing-import]
from geopy.geocoders import Nominatim
import time

blocks = [
    "Hilsa", "Islampur", "Biharsharif", "Rajgir", "Harnaut", "Noorsarai",
    "Rahui", "Asthawan", "Chandi", "Ekangarsarai", "Bind", "Silao",
    "Giriyak", "Tharthari", "Karai Parsurai", "Katrisarai", "Ben",
    "Sarmera", "Parbalpur", "Warisaliganj"
]

geolocator = Nominatim(user_agent="nalanda_app")
results = {}

for block in blocks:
    try:
        # Append Nalanda, Bihar to ensure accuracy
        query = f"{block}, Nalanda, Bihar, India"
        if block == "Warisaliganj":
            query = f"{block}, Nawada, Bihar, India" # since it's actually in Nawada
        
        location = geolocator.geocode(query)
        if location:
            results[block] = {"lat": location.latitude, "lon": location.longitude}
            print(f"Found {block}: {location.latitude}, {location.longitude}")
        else:
            # Fallback
            location = geolocator.geocode(f"{block}, Bihar, India")
            if location:
                results[block] = {"lat": location.latitude, "lon": location.longitude}
                print(f"Found {block} (fallback): {location.latitude}, {location.longitude}")
            else:
                print(f"Not found: {block}")
    except Exception as e:
        print(f"Error for {block}: {e}")
    time.sleep(1)

with open("block_coords.json", "w") as f:
    json.dump(results, f, indent=4)
