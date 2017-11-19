'''
Weather simulation:

This simulator will provide the function to generate weather data for a game.

Requirement:
This simulator require some modules to run, which can be installed as below:
pip install requests
pip install numpy
pip install sklearn

Usage:  python simulator.py

'''

import os, sys, errno
import datetime
import csv
import requests
import numpy
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB


class NaiveBayesClassifier_Linear(linear_model.LinearRegression):
    '''
    Implementation of Naive Bayes classifier algorithm. In this case,
    I chose to use scikit module for Naive Bayes classifier (credit on scikit), and
    encapsulate here.
    '''

    def __init__(self):
        super(NaiveBayesClassifier_Linear, self).__init__()

    def fit(self, x, y):
        return super(NaiveBayesClassifier_Linear, self).fit(x, y)

    def predict(self, x):
        return super(NaiveBayesClassifier_Linear, self).predict(x)



class NaiveBayesClassifier_GNB(GaussianNB):
    '''
    Implementation of Naive Bayes classifier algorithm. In this case,
    I chose to use scikit module for Naive Bayes classifier (credit on scikit), and
    encapsulate here.
    '''

    def __init__(self):
        super(NaiveBayesClassifier_GNB, self).__init__()

    def fit(self, x, y):
        return super(NaiveBayesClassifier_GNB, self).fit(x, y)

    def predict(self, x):
        return super(NaiveBayesClassifier_GNB, self).predict(x)


class CSVOperator:
    '''
    Manipulating CSV data and files
    '''

    write_fieldnames = ["Location", "Position", "Local Time",
                                 "Conditions", "Temperature", "Pressure", "Humidity"]

    def load_csv(self, file_path="", delimiter=',', fieldnames=[]):
        if os.path.exists(file_path) and os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                if fieldnames:
                    reader = csv.DictReader(f, delimiter=delimiter, fieldnames=fieldnames)
                else:
                    reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    yield row
                f.close()
        else:
            raise FileExistsError("File does not exist!")


    def write_csv(self, data=None, path="data/", name="weather_data.csv", delimiter="|", fieldnames=[]):
        if data:
            if not os.path.exists(path):
                try:
                    os.makedirs(path)
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise
                    pass

            with open(path+name, 'w') as f:
                writer = csv.DictWriter(f, delimiter=delimiter,
                                        fieldnames=fieldnames if fieldnames else self.write_fieldnames,
                                        extrasaction='ignore')
                if fieldnames:
                    writer.writeheader()
                writer.writerows(data)
                f.close()


class DataRetriever:
    '''
    Retrieving training data from Google Geography API for geography data,
    Google Elevation API for elevation data and Dark Sky API for weather data.
    '''

    def __init__(self):
        self.google_geography_api = "http://maps.googleapis.com/maps/api/geocode/json?sensor=false&address="
        self.google_elevation_api = "https://maps.googleapis.com/maps/api/elevation/json?locations="
        self.darksky_weather_api = "https://api.darksky.net/forecast/0123456789abcdef9876543210fedcba/"

    def retrieve_json_data(self, url):
        """
        download json data from api
        """
        try:
            res = requests.get(url).json()
            return res
        except requests.exceptions.RequestException as err:
            print(err)
            sys.exit(1)

    def get_location_names(self):
        """
        locations for retrieving geography data
        """
        loc_dict = CSVOperator().load_csv('data/locations.csv')
        return [loc.get('location','') for loc in loc_dict]

    def get_location_data_list(self):
        """
        retrieve geography data and store it in list
        """
        locations = self.get_location_names()
        loc_data_list = []
        for location in locations:

            loc_data = {'location': location}

            # retrieve geography data
            res_of_geo = self.retrieve_json_data(self.google_geography_api+location)
            if res_of_geo and res_of_geo.get('results', None):
                # parse return geography data
                for geo_item in res_of_geo.get('results', {}):
                    # parse lat and lng data
                    lat_lng = geo_item.get('geometry', '').get('location', '')
                    loc_data['lat'] = lat_lng.get('lat','')
                    loc_data['lng'] = lat_lng.get('lng', '')

                    # retrieve elevation data
                    res_of_elev = self.retrieve_json_data(self.google_elevation_api + str(loc_data.get('lat','')) + ',' + str(loc_data.get('lng','')))
                    if res_of_elev and res_of_elev.get('results', None):
                        # parse return elevation data
                        for elev_item in res_of_elev.get('results', {}):
                            # parse elevation
                            loc_data['elevation'] = elev_item.get('elevation','')

            # add to location data list
            if set(['elevation','lat','lng','location']) == set(list(loc_data.keys())):
                loc_data_list.append(loc_data)
        return loc_data_list

    def get_weather_data(self, days=1):

        location_data_list = self.get_location_data_list()

        # for storing weather data
        wd_list = []
        wd_keys = ['condition','temperature','pressure','humidity','time','location','lat','lng','elevation']

        # get weather data X days ago from now, here default set x = 60
        day_peroid = [datetime.datetime.now() - datetime.timedelta(day) for day in range(1, days+1)]

        # retrieve weather data from api by location data
        for loc_data in location_data_list:
            for day in day_peroid:
                res_of_wd = self.retrieve_json_data(self.darksky_weather_api + "%s,%s,%s" % (loc_data.get('lat', ''),
                                                                                             loc_data.get('lng', ''),
                                                                                             day.strftime('%s')))

                if res_of_wd and res_of_wd.get('hourly', None):
                    # parse hourly weather data
                    for data in res_of_wd.get('hourly', {}).get('data', []):
                        if set(wd_keys).issubset(set(data.keys)):
                            wd = {}
                            wd['condition'] = data.get('summary', '')
                            wd['temperature'] = data.get('temperature', 30)
                            wd['pressure'] = data.get('pressure', 1000)
                            wd['humidity'] = data.get('humidity', 0.3)
                            wd['time'] = data.get('time', '')
                            wd['location'] = loc_data.get('location', '')
                            wd['lat'] = loc_data.get('lat', '')
                            wd['lng'] = loc_data.get('lng', '')
                            wd['elevation'] = loc_data.get('elevation', '')
                            wd_list.append(wd)

        return wd_list


class Simulator:
    '''
    Implementation of simulating weather and generating weather data through steps such as:
    training weather data, predict weather data, generate weather data etc.
    '''

    def __init__(self):
        self.data_retriever = DataRetriever()
        self.csv_operator = CSVOperator()
        self.nbcg = NaiveBayesClassifier_GNB
        self.nbcl = NaiveBayesClassifier_Linear

    def create_array(self, elements, dataset):
        if elements and dataset:
            _list = []
            for data in dataset:
                tmp_list = []
                for element in elements:
                    if element == 'condition':
                        tmp_list.append(data[element])
                    elif element in ['temperature','pressure','humidity']:
                        tmp_list.append(float(data[element]))
                    else:
                        tmp_list.append(float(data[element]))


                _list.append(tmp_list)

            if elements[0] == 'condition':
                return numpy.array(_list, dtype=object)
            else:
                return numpy.array(_list)
        return []

    def train_weather_data(self, file_path="data/train_data.csv"):
        """
        train the weather data using Naive Bayes classifier to predict weather data
        """

        if os.path.exists(file_path) and os.path.isfile(file_path):
            train_data = [row for row in self.csv_operator.load_csv(file_path)]
        else:
            # the default is to retrieve 1 day ago weather data, if you want to retrieve
            # more, simplely add days parameter to get_weather_data() method, e.g.
            # self.data_retriever.get_weather_data(days=7), please delete the train_data.csv
            # for updating train_data
            train_data = self.data_retriever.get_weather_data()
            self.csv_operator.write_csv(data=train_data, name="train_data.csv", delimiter=',',
                                        fieldnames=list(next(train_data).keys()))
        # geography and elevation data for training weather data
        x_para = self.create_array(['elevation','lat','lng','time'],train_data)

        # training temperature
        self.temperature_model = self.nbcl()
        self.temperature_model.fit(x_para, self.create_array(['temperature',],train_data))

        # training pressure
        self.pressure_model = self.nbcl()
        self.pressure_model.fit(x_para, self.create_array(['pressure',], train_data))

        # training humidity
        self.humidity_model = self.nbcl()
        self.humidity_model.fit(x_para, self.create_array(['humidity',], train_data))

        # training conditions
        self.condition_model = self.nbcg()
        self.condition_model.fit(x_para, self.create_array(['condition',], train_data))

    def generate_weather_data(self):
        weather_data_list = []

        # 1. train the past weather data
        print("1. training past weather data...")
        self.train_weather_data()

        # 2. get the geography and elevation data of the cities which are going to predicted weather,
        # in this case will use the same cities with training data, because here is training the past
        # data to predict today's weather, and the time is now
        print("2. get location data...")
        location_data_list = self.data_retriever.get_location_data_list()
        time = datetime.datetime.now()

        # 3. get predicted weather data of each city
        print("3. generating weather data")
        for loc_data in location_data_list:
            wd = {}
            # geography and elevation data for predicting weather data
            x_para = numpy.array([float(loc_data['elevation']), float(loc_data['lat']), float(loc_data['lng']), float(time.strftime('%s'))]).reshape(1,4)
            wd['Location'] = loc_data.get('location', '')
            wd['Position'] = "%.2f,%.2f,%d" % (loc_data.get('lat',''), loc_data.get('lng',''), loc_data.get('elevation', ''))
            wd['Local Time'] = time.isoformat()

            # get predict weather data
            temp = (self.temperature_model.predict(x_para)[0][0] - 32) * 5.0 / 9.0
            if temp >= 0:
                wd['Temperature'] = '+%.1f' % temp
            else:
                wd['Temperature'] = '-%.1f' % temp
            wd['Pressure'] = '%.1f' % self.pressure_model.predict(x_para)[0][0]
            wd['Humidity'] = '%d' % (self.humidity_model.predict(x_para)[0][0] * 100)
            condition = self.condition_model.predict(x_para)[0].lower()
            if 'snow' in condition:
                wd['Conditions'] = 'Snow'
            elif 'clear' in condition:
                wd['Conditions'] = 'Sunny'
            else:
                wd['Conditions'] = 'Rain'
            weather_data_list.append(wd)

        # 4. write to csv file weather_data.csv
        self.csv_operator.write_csv(weather_data_list)

    def run(self):
        print("Start simulation...")
        self.generate_weather_data()
        print("End simulation...")


simulator = Simulator()
simulator.run()