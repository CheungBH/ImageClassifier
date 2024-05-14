from simple_image_download import simple_image_download as sp 

response = sp.Downloader
response.extensions = {'.jpg', '.jpeg'}

moon_phases = [#'Angkor Wat: Siem Reap', 
             #'Sydney Opera House: Sydney', 
             "nile river scenery", 
             "nile river photography", 
             "nile river real"
             # 'Mount Rushmore National Memorial: Keystone',
             #'Mont-Saint-Michel: Normandy', 'The Acropolis: Athens'
             #'The Brandenburg Gate: Berlin', 'Easter Island: Chile', 
             #'Golden Gate Bridge: San Francisco', 'Neuschwanstein Castle: Schwangau', 
             #Leaning Tower of Pisa: Pisa', 'The Great Pyramid of Giza: Giza',
             #'Victoria Falls: Zimbabwe and Zambia', 'Buckingham Palace: London',
             #'Basilica de la Sagrada Familia: Barcelona', 'Christ the Redeemer: Rio de Janeiro',
             #'Blue Mosque: Istanbul', 'The Colosseum: Rome', 'The Grand Palace: Bangkok', 
             #'Statue of Liberty: New York City', 'Petra: Wadi Musa', 'Ha Long Bay: Ha Long',
             #'Stonehenge: Salisbury', 'Blue Domes of Oia: Santorini', 
             #'Mount Fuji: Honshu', 'Potala Palace: Lhasa', 
             #'Lake Louise: Banff National Park', 'Grand Canyon: Grand Canyon National Park'
             #'Uyuni Salt Flats: Bolivia'
            ]

for i in moon_phases:
    counter = 0
    while counter != 4:
        response = sp.Downloader
        response.extensions = {'.jpg', '.jpeg'}
        response().download(keywords=str(i + "_" + str(counter)), limit=500)
        counter += 1
