from .config import *

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime
import dateutil.relativedelta
import osmnx as ox
import geopandas as gpd
from geopandas.tools import sjoin
import mlai
import mlai.plot as plot
from sklearn import preprocessing
from sklearn.decomposition import PCA


"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

"""The following functions mostly have to deal with cleaning data"""


def one_hot(df, col_prefix, col):
  """One hot encode a given column col"""
  one_hot = pd.get_dummies(df[col])
  df = df.drop(col, axis=1)
  columns = [f"{col_prefix}_{i}" for i in one_hot.columns]
  one_hot.columns = columns
  df = df.join(one_hot)
  return df


def bin_price(df):
  y = np.array(df['price'])
  y_1 = np.sort(y)
  buckets = [np.percentile(y_1, i) for i in range(0, 100, 10)]
  y = np.digitize(y, buckets)
  df['price'] = y
  return df


def scale_and_reduce(df):
  """For PCA, some columns need to be scaled"""
  cols_to_keep = list(df.columns)
  cols_to_keep.remove('price')
  df_1 = df[cols_to_keep]
  scaled = preprocessing.scale(df_1)
  df_1_s = pd.DataFrame(scaled, columns=cols_to_keep)
  df = df.drop(cols_to_keep, axis=1)
  return df.join(df_1_s)


"""The following functions deal with finding specific data"""

def get_lat_and_long_box(df):
  lat_min, lat_max = min(df['latitude']), max(df['latitude'])
  lon_min, lon_max = min(df['longitude']), max(df['longitude'])
  return float(lat_max), float(lat_min), float(lon_max), float(lon_min)


def get_mean_lat_and_long(df):
  latitude, longitude = np.mean(df['latitude']), np.mean(df['longitude'])
  return latitude, longitude


def filter_to_only_data_within_box(df, latitude, longitude, bounds):
  df = df[(df['longitude'] < float(longitude) + bounds) & (df['longitude'] > float(longitude) - bounds)]
  df = df[(df['latitude'] < float(latitude) + bounds) & (df['latitude'] > float(latitude) - bounds)]
  return df.reset_index(drop = True)


def find_postcode(df, longitude, latitude, bounds=0.001):
  # For some reason, using the apply function on pandas has an import error and I don't have the time to debug it
  # so working around it
  df = filter_to_only_data_within_box(df, latitude, longitude, bounds)

  def euc_dis(row):
    return (float(row['longitude']) - float(longitude))**2 + (float(row['latitude']) - float(latitude))**2

  df['distance'] = [euc_dis(row) for _, row in df.iterrows()]
  if len(df) == 0:
    raise Exception("Check your longitude and latitudes. They might not correspond to a location in the UK")
  df = df.sort_values(by = ['distance'])
  return df['postcode'].tolist()[0]

"""The following functions have to do with viewing data"""


def view_poi_map(town_city, latitude, longitude, diff_lat, diff_long, tags):
  """View of place of interest for a given town city"""
  box_width = diff_lat
  box_height = diff_long
  north = latitude + box_height / 2
  south = latitude - box_width / 2
  west = longitude - box_width / 2
  east = longitude + box_width / 2
  pois = ox.geometries_from_bbox(north, south, east, west, tags)

  graph = ox.graph_from_bbox(north, south, east, west)
  nodes, edges = ox.graph_to_gdfs(graph)
  area = ox.geocode_to_gdf(town_city)

  fig, ax = plt.subplots(figsize=plot.big_figsize)

  # Plot the footprint
  area.plot(ax=ax, facecolor="white")

  # Plot street edges
  edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

  ax.title(f"Map of {town_city}")
  ax.set_xlim([west, east])
  ax.set_ylim([south, north])
  ax.set_xlabel("longitude")
  ax.set_ylabel("latitude")

  # Plot all POIs
  pois.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
  plt.tight_layout()


def view_pois_and_df_map(town_city, df, tags, price_bin):
  """View of place of interest with scatter plot of house prices for a given town city"""
  df2 = df.reset_index(drop=True)

  north, south, east, west = get_lat_and_long_box(df)

  pois = ox.geometries_from_bbox(north, south, east, west, tags)

  graph = ox.graph_from_bbox(north, south, east, west)
  nodes, edges = ox.graph_to_gdfs(graph)
  area = ox.geocode_to_gdf(town_city)

  fig, ax = plt.subplots(figsize=plot.big_figsize)

  # Plot the footprint
  area.plot(ax=ax, facecolor="white")

  # Plot street edges
  edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

  ax.title(f"Map of {town_city}")
  ax.set_xlim([west, east])
  ax.set_ylim([south, north])
  ax.set_xlabel("longitude")
  ax.set_ylabel("latitude")

  # Plot all POIs
  pois.plot(ax=ax, color="blue", alpha=0.7, markersize=10)

  colors = cm.rainbow(np.linspace(0, 1, 10))
  for price_bucket in range(1, 11):
    df_1 = df2[df2['price'] == price_bucket]
    df_1 = gpd.GeoDataFrame(df_1, geometry=gpd.points_from_xy(df_1['longitude'], df_1['latitude']))
    df_1.plot(ax=ax, color=colors[price_bucket - 1], alpha=1, markersize=75)

  plt.tight_layout()


def view_price(df, all_cols = False):
  """View price against a single column"""
  col = None
  if all_cols:
    for col in df.columns:
      plt.figure()
      plt.title(f"Graph of {col} against price")
      plt.scatter(df[col], df['price'])
      plt.show()
  else:
    while col not in df.columns:
      # Using this loop since I don't really want to do too much error handling
      print(df.columns)
      col = input("Enter the column name you want to view price against")
    plt.figure()
    plt.title(f"Graph of {col} against price")
    plt.scatter(df[col], df['price'])
    plt.show()


def view_pca(df):
  """Generates a 1d,2d or 3d scatter plot of attributes with respect to price"""
  dim = -1
  while dim not in {1, 2, 3}:
    # Using this loop since I don't really want to do too much error handling
    dim = int(input("Select a number 1 to 3 which corresponds to the number of dimensions you want to see the data in"))
  df = bin_price(df)
  pca = PCA(n_components=dim)
  useful_cols = list(df.columns)
  useful_cols.remove('price')
  x = df[useful_cols]
  y = df['price']
  df = pd.DataFrame(pca.fit_transform(x))
  df = df.join(y)

  colors = cm.rainbow(np.linspace(0, 1, 10))
  if dim == 1:
    plt.scatter(df[0], y)

  elif dim == 2:
    for index, row in df.iterrows():
      plt.scatter(row[0], row[1], color=colors[int(row['price']) - 1])
    print("Red represents the most expensive 10% house in the area, violet represents the least expensive 10%.")

  elif dim == 3:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for _, row in df.iterrows():
      ax.scatter(row[0], row[1], row[2], color=colors[int(row['price']) - 1])
    plt.show()
    print("Red represents the most expensive 10% house in the area, violet represents the least expensive 10%.")


def view_map(df):
  """Generates a 2d map of the area including pois and the cost of the houses"""
  town_city = df['town_city'].unique().item()
  possible_tags = {"amenity": 0, "buildings": 0, "historic": 0, "leisure": 0, "shop": 0, "tourism": 0}
  for tag in possible_tags:
    value = bool(input(f"View tag {tag}? 1 for yes, 0 for no"))
    possible_tags[tag] = value
  price_bin = bin_price(df)
  view_pois_and_df_map(town_city, df, possible_tags, price_bin)


"""The following functions mostly have to deal with adding features to the data"""

def count_nearby_pois(latitude, longitude, poi_tag, bounds, pois):
  if poi_tag not in pois.columns:
    return 0
  pois = pois[pois[poi_tag].notnull()]
  pois = filter_to_only_data_within_box(pois, latitude, longitude, bounds)
  return len(pois)


def add_pois(place_name, df, poi_tags):
  """Given a place_name and tags, for each poi_tag find the number of pois in the vicinity of each house in df"""
  df = df.loc[df.town_city == place_name].reset_index(drop=True)
  north, south, east, west = get_lat_and_long_box(df)
  pois = ox.geometries_from_bbox(north, south, east, west, poi_tags)
  pois = pois.xs('node').reset_index(drop = True)
  lat_and_long = [row.geometry.representative_point().coords[:][0] for _, row in pois.iterrows()]
  lat_and_long = pd.DataFrame(lat_and_long, columns=['longitude', 'latitude'])
  pois = pd.concat([pois, lat_and_long], axis=1)
  for poi_tag in poi_tags:
    df[poi_tag] = [count_nearby_pois(row['latitude'], row['longitude'], poi_tag, 0.02, pois) for _, row in
                   df.iterrows()]
  return df


def add_statistics_of_houses_sold_before(df):
  """Find the statistics of houses sold in a given location to check the immediate demand"""

  def get_number(row, df=df, months_prior=1):
    area, d1 = row['postcode'], row['date_of_transfer']
    d2 = d1 - dateutil.relativedelta.relativedelta(months=months_prior)
    df = df[df['postcode'] == area]
    total = len(df)
    number_sold_before = df[(df['date_of_transfer'] < d1) & (df['date_of_transfer'] > d2)]
    sold_before = len(number_sold_before)
    average_price = np.mean(df['price'])
    return sold_before, total, average_price

  stats = [get_number(row) for _, row in df.iterrows()]
  df1 = pd.DataFrame(stats, columns=['sold_before', 'sold_total', 'average_price_of_area'])
  df = pd.concat([df, df1], axis=1)
  df.fillna(0)
  return df


def add_inverse_of_columns(df, col_lst):
  inverse_col_lst = [f"inv_{tag}" for tag in col_lst]
  for i_key, key in zip(inverse_col_lst, col_lst):
    df[i_key] = 1 / (1 + df[key])
  return df


def add_one_hot_property_type(df, one_hot_cols):
  if len(df['property_type'].unique()) < 5:
    property_dct = {'D': [1, 0, 0, 0, 0],
                    'F': [0, 1, 0, 0, 0],
                    'O': [0, 0, 1, 0, 0],
                    'S': [0, 0, 0, 1, 0],
                    'T': [0, 0, 0, 0, 1]}
    P_df = [property_dct[row['property_type']] for _, row in df.iterrows()]
    df = df.drop(columns=['property_type'])
    P_df = pd.DataFrame(P_df, columns = one_hot_cols)
    df = pd.concat([df, P_df], axis = 1)
  else:
    df = one_hot(df, 'pt', 'property_type')
  return df


def add_postcode_number(df):
  postcode_data = [int(row['postcode'][row['postcode'].index(' '): -2]) for _, row in df.iterrows()]
  df['postcode'] = postcode_data
  return df


"""Standard assess functions"""

def data():
  """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
  df = access.data()
  df.fillna(0)
  return df


def query(data):
  return


def view(data) -> None:
  """Provide a view of the data that allows the user to verify some aspect of its quality."""
  view_type = -1
  while view_type not in {0, 1, 2}:
    # Using this loop since I don't really want to do too much error handling
    view_type = int(input("""Enter 0 to view price with respect to a specific column in df, 1 to view price as as PCA scatter plot
                         2 to view a scatter plot map of house prices in a given town"""))
  f = {0: view_price, 1: view_pca, 2: view_map}
  f[view_type](data)


def labelled(df, town_city, poi_tags):
  """Provide a labelled set of data ready for supervised learning."""
  # Feature that I want to include unfortunately have to be hard coded in

  required_cols = ['price', 'latitude', 'longitude']
  property_one_hot_cols = ['pt_D', 'pt_F', 'pt_O', 'pt_S', 'pt_T']
  house_stats_cols =  ['sold_before', 'sold_total']
  inverse_house_stats_cols  = [f"inv_{tag}" for tag in house_stats_cols]
  inverse_poi_tags = [f"inv_{tag}" for tag in poi_tags]

  df = add_pois(town_city, df, poi_tags)
  df = add_inverse_of_columns(df, poi_tags)
  df = add_inverse_of_columns(df, house_stats_cols)
  df = add_statistics_of_houses_sold_before(df)
  df = add_one_hot_property_type(df, property_one_hot_cols)

  columns = required_cols + property_one_hot_cols + inverse_house_stats_cols + house_stats_cols + inverse_poi_tags
  return df[columns]
