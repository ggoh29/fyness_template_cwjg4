from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import mongodb
import sqlite"""

import yaml
from ipywidgets import interact_manual, Text, Password
import pymysql
import pandas as pd
# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

"""The following functions are related to database connections"""

@interact_manual(username=Text(description="Username:"),password=Password(description="Password:"))
def write_credentials(username, password):
  with open("credentials.yaml", "w") as file:
    credentials_dict = {'username': username,
                        'password': password}
    yaml.dump(credentials_dict, file)

def create_connection(user, password, host, database, port=3306):
  """ Create a database connection to the MariaDB database
      specified by the host url and database name.
  :param user: username
  :param password: password
  :param host: host url
  :param database: database
  :param port: port number
  :return: Connection object or None
  """
  conn = None
  try:
    conn = pymysql.connect(user=user,
                           passwd=password,
                           host=host,
                           port=port,
                           local_infile=1,
                           db=database
                           )
  except Exception as e:
    print(f"Error connecting to the MariaDB Server: {e}")
  return conn


def create_conn(database):
  database_details = {"url": "database-1.cx4sotafoi1m.eu-west-2.rds.amazonaws.com",
                      "port": 3306}

  with open("credentials.yaml") as file:
    credentials = yaml.safe_load(file)
  username = credentials["username"]
  password = credentials["password"]
  url = database_details["url"]

  conn = create_connection(user=credentials["username"],
                                 password=credentials["password"],
                                 host=database_details["url"],
                                 database=database)
  return conn

"""The following functions are helper functions for accessing speciic aspects of the data"""

def _access_postcode_data(conn):
  cur = conn.cursor()
  cur.execute("""SELECT longitude, lattitude, postcode FROM postcode_data;""")
  rows = cur.fetchall()
  cols = ['longitude', 'latitude', 'postcode']
  return pd.DataFrame(rows, columns=cols)


def _access_house_prices(conn):
  cur = conn.cursor()

  cur.execute("""SELECT price, date_of_transfer, property_type, new_build_flag, 
     tenure_type, locality, town_city, district, 
     county, db_id, postcode FROM pp_data_2""")
  row = cur.fetchall()

  cols = ['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type', 'locality', 'town_city', 'district', 'county', 'db_id']
  return pd.DataFrame(row, columns=cols)


def _access_house_prices_by_year(year, conn):
  cur = conn.cursor()

  cur.execute(f"""SELECT price, date_of_transfer, postcode, property_type, new_build_flag, 
     tenure_type, locality, town_city, district, 
     county, db_id FROM pp_data_2
     WHERE date_of_transfer >= '{year}-01-01 00:00:00'
       AND date_of_transfer < '{year + 1}-01-01 00:00:00'""")
  row = cur.fetchall()

  cols = ['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type', 'locality', 'town_city', 'district', 'county', 'db_id']
  return pd.DataFrame(row, columns=cols)


def _access_house_prices_by_year_and_town_city(year, city, conn):
  cur = conn.cursor()

  cur.execute(f"""SELECT price, date_of_transfer, postcode, property_type, new_build_flag, 
     tenure_type, locality, town_city, district, 
     county, db_id FROM pp_data_2
     WHERE date_of_transfer >= '{year}-01-01 00:00:00'
       AND date_of_transfer < '{year + 1}-01-01 00:00:00'
       AND town_city = '{city}'""")
  row = cur.fetchall()

  cols = ['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type', 'locality', 'town_city', 'district', 'county', 'db_id']
  return pd.DataFrame(row, columns=cols)


def _access_towncity_from_postcode(postcode, conn):
  cur = conn.cursor()

  cur.execute(f"""SELECT town_city, postcode FROM pp_data_2
     WHERE postcode = '{postcode}'""")
  row = cur.fetchall()

  cols = ['town_city', 'postcode']
  df = pd.DataFrame(row, columns=cols)
  return df['town_city'].unique()[0]

"""The following are function for accessing data but intended as api calls so these are the functions the client sees"""

def access_merged_data_by_year(year : int):

  house_conn = create_conn("house_prices")
  house_prices = _access_house_prices_by_year(year, house_conn)

  postcode_conn = create_conn("property_prices")
  property_prices = _access_postcode_data(postcode_conn)
  return pd.merge(house_prices, property_prices, on='postcode', how='inner')


def access_merged_data_by_year_and_town_city(year : int, town_city : str):

  house_conn = create_conn("house_prices")
  house_prices = _access_house_prices_by_year_and_town_city(year, town_city, house_conn)

  postcode_conn = create_conn("property_prices")
  property_prices = _access_postcode_data(postcode_conn)
  return pd.merge(house_prices, property_prices, on='postcode', how='inner')


def access_raw_postcodes():
  postcode_conn = create_conn("property_prices")
  property_prices = _access_postcode_data(postcode_conn)
  return property_prices


def access_town_given_postcode(postcode):
  house_conn = create_conn("house_prices")
  return _access_towncity_from_postcode(postcode, house_conn)


def query():
  database = input('Which database do you want to use?')
  q = input('Insert an SQL query')

  conn = create_conn(database)
  cur = conn.cursor()
  cur.executy(q)
  return cur.fetchall()


def data():
  """Read the data from the web or local file, returning structured format such as a data frame"""

  house_conn = create_conn("house_prices")
  house_prices = _access_house_prices(house_conn)

  postcode_conn = create_conn("property_prices")
  property_prices = _access_postcode_data(postcode_conn)

  return pd.merge(house_prices, property_prices, on = 'postcode', how = 'inner')