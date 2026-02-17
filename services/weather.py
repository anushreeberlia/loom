"""
Weather service using OpenWeatherMap API.
"""

import os
import httpx
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


@dataclass
class WeatherData:
    """Structured weather information for outfit recommendations."""
    temperature_c: float  # Celsius
    temperature_f: float  # Fahrenheit
    feels_like_c: float
    feels_like_f: float
    humidity: int  # percentage
    wind_speed: float  # m/s
    condition: str  # main condition: Clear, Clouds, Rain, Snow, etc.
    description: str  # detailed description
    icon: str  # weather icon code
    city: str
    
    @property
    def is_cold(self) -> bool:
        """Below 15°C / 59°F - suggest layers"""
        return self.feels_like_c < 15
    
    @property
    def is_hot(self) -> bool:
        """Above 25°C / 77°F - light clothes"""
        return self.feels_like_c > 25
    
    @property
    def is_rainy(self) -> bool:
        return self.condition.lower() in ["rain", "drizzle", "thunderstorm"]
    
    @property
    def is_snowy(self) -> bool:
        return self.condition.lower() == "snow"
    
    @property
    def suggested_season(self) -> str:
        """Map temperature to season tag for filtering."""
        if self.feels_like_c < 10:
            return "winter"
        elif self.feels_like_c < 18:
            return "fall"  # or spring
        elif self.feels_like_c < 25:
            return "spring"
        else:
            return "summer"
    
    def to_dict(self) -> dict:
        return {
            "temperature_c": round(self.temperature_c, 1),
            "temperature_f": round(self.temperature_f, 1),
            "feels_like_c": round(self.feels_like_c, 1),
            "feels_like_f": round(self.feels_like_f, 1),
            "humidity": self.humidity,
            "wind_speed": round(self.wind_speed, 1),
            "condition": self.condition,
            "description": self.description,
            "icon": self.icon,
            "icon_url": f"https://openweathermap.org/img/wn/{self.icon}@2x.png",
            "city": self.city,
            "is_cold": self.is_cold,
            "is_hot": self.is_hot,
            "is_rainy": self.is_rainy,
            "is_snowy": self.is_snowy,
            "suggested_season": self.suggested_season,
        }


async def fetch_weather(lat: float, lon: float) -> Optional[WeatherData]:
    """
    Fetch current weather from OpenWeatherMap.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        WeatherData object or None if failed
    """
    if not OPENWEATHER_API_KEY:
        logger.warning("OPENWEATHER_API_KEY not set")
        return None
    
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            
        if response.status_code != 200:
            logger.error(f"Weather API error: {response.status_code} - {response.text}")
            return None
        
        data = response.json()
        
        temp_c = data["main"]["temp"]
        feels_like_c = data["main"]["feels_like"]
        
        weather = WeatherData(
            temperature_c=temp_c,
            temperature_f=temp_c * 9/5 + 32,
            feels_like_c=feels_like_c,
            feels_like_f=feels_like_c * 9/5 + 32,
            humidity=data["main"]["humidity"],
            wind_speed=data["wind"]["speed"],
            condition=data["weather"][0]["main"],
            description=data["weather"][0]["description"],
            icon=data["weather"][0]["icon"],
            city=data.get("name", "Unknown"),
        )
        
        logger.info(f"Weather for {weather.city}: {weather.temperature_c}°C, {weather.condition}")
        return weather
        
    except Exception as e:
        logger.error(f"Weather fetch error: {e}")
        return None


def get_weather_outfit_adjustments(weather: WeatherData) -> dict:
    """
    Get outfit generation adjustments based on weather.
    
    Returns dict with:
        - force_layer: bool - always include layer slot
        - skip_layer: bool - don't include layer slot  
        - preferred_seasons: list - season tags to boost
        - avoid_seasons: list - season tags to penalize
        - notes: list - human-readable suggestions
    """
    adjustments = {
        "force_layer": False,
        "skip_layer": False,
        "preferred_seasons": ["all-season"],
        "avoid_seasons": [],
        "notes": [],
    }
    
    # Temperature-based
    if weather.is_cold:
        adjustments["force_layer"] = True
        adjustments["preferred_seasons"].extend(["winter", "fall"])
        adjustments["avoid_seasons"].append("summer")
        adjustments["notes"].append(f"Cold weather ({weather.feels_like_c:.0f}°C) - adding layers")
    elif weather.is_hot:
        adjustments["skip_layer"] = True
        adjustments["preferred_seasons"].extend(["summer", "spring"])
        adjustments["avoid_seasons"].append("winter")
        adjustments["notes"].append(f"Hot weather ({weather.feels_like_c:.0f}°C) - light clothes recommended")
    else:
        adjustments["preferred_seasons"].extend(["spring", "fall"])
        adjustments["notes"].append(f"Moderate weather ({weather.feels_like_c:.0f}°C)")
    
    # Precipitation-based
    if weather.is_rainy:
        adjustments["notes"].append("Rainy - consider waterproof layer")
        adjustments["force_layer"] = True  # Rain jacket
    
    if weather.is_snowy:
        adjustments["notes"].append("Snowy - warm layers and boots recommended")
        adjustments["force_layer"] = True
        adjustments["preferred_seasons"] = ["winter"]
    
    return adjustments


def get_occasion_from_time() -> dict:
    """
    Auto-detect appropriate occasion based on current day and time.
    
    Returns dict with:
        - occasion: str - primary occasion tag to prefer
        - avoid_occasions: list - occasions to deprioritize
        - note: str - human-readable suggestion
    """
    from datetime import datetime
    
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    
    is_weekend = weekday >= 5  # Saturday or Sunday
    
    if is_weekend:
        if hour >= 18:  # Evening
            return {
                "occasion": "going-out",
                "prefer_occasions": ["going-out", "date", "party", "dinner"],
                "avoid_occasions": ["work", "office", "formal"],
                "note": "Weekend evening - going out vibes 🎉"
            }
        else:  # Daytime weekend
            return {
                "occasion": "casual",
                "prefer_occasions": ["casual", "brunch", "everyday", "relaxed"],
                "avoid_occasions": ["work", "office", "formal"],
                "note": "Weekend - keep it casual 😎"
            }
    else:
        # Weekday
        if hour < 17:  # Before 5 PM
            return {
                "occasion": "work",
                "prefer_occasions": ["work", "office", "business-casual", "professional"],
                "avoid_occasions": ["party", "clubbing", "beach"],
                "note": "Workday - office appropriate 💼"
            }
        elif hour < 20:  # 5-8 PM - transitional
            return {
                "occasion": "smart-casual",
                "prefer_occasions": ["smart-casual", "dinner", "after-work", "date"],
                "avoid_occasions": ["beach", "gym", "loungewear"],
                "note": "Evening - smart casual for after-work 🍷"
            }
        else:  # After 8 PM
            return {
                "occasion": "going-out",
                "prefer_occasions": ["going-out", "dinner", "date", "night-out"],
                "avoid_occasions": ["work", "office", "gym"],
                "note": "Night out - dress to impress ✨"
            }

