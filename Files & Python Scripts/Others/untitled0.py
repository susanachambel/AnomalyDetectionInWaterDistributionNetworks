# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:13:26 2019

@author: susan
"""

"""
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="specify_your_app_name_here")
location = geolocator.reverse("52.509669, 13.376294")
print(location.address)
print((location.latitude, location.longitude))
print(location.raw)
"""

from geopy.geocoders import GoogleV3
geolocator = GoogleV3(api_key='AIzaSyCfa-2sC_tnxCp6gO0F0snaKYMEGEwVq2Q')
location = geolocator.reverse("52.509669, 13.376294")
print(location.address)