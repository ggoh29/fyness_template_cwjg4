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


def get_lat_and_long_box(df):
    lat_min, lat_max = min(df['latitude']), max(df['latitude'])
    lon_min, lon_max = min(df['longitude']), max(df['longitude'])
    return float(lat_max), float(lat_min), float(lon_max), float(lon_min)


def get_statistics_of_houses_sold_before(df):
    """Find the statistics of houses sold in a given location to check the immediate demand"""
    def get_number(row, df=df, months_prior = 1):
        area, date = row['postcode'], row['date_of_transfer']
        d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M")
        d2 = d - dateutil.relativedelta.relativedelta(months=months_prior)
        d1, d2 = str(d), str(d2)
        df = df[df['postcode'] == area]
        total = len(df)
        number_sold_before = df[(df['date_of_transfer'] < d1) & (df['date_of_transfer'] > d2)]
        sold_before = len(number_sold_before)
        average_price = np.mean(df['price'])
        return sold_before, total, average_price

    df[['sold_before', 'sold_total', 'average_price_of_area']] = pd.DataFrame(df.apply(get_number, axis=1).tolist())
    df.fillna(0)
    return df

def get_poi_map(town_city, latitude, longitude, diff_lat, diff_long, tags):
    """View of place of interest for a given town city"""
    box_width = diff_lat
    box_height = diff_long
    north = latitude + box_height/2
    south = latitude - box_width/2
    west = longitude - box_width/2
    east = longitude + box_width/2
    pois = ox.geometries_from_bbox(north, south, east, west, tags)

    graph = ox.graph_from_bbox(north, south, east, west)
    nodes, edges = ox.graph_to_gdfs(graph)
    area = ox.geocode_to_gdf(town_city)

    fig, ax = plt.subplots(figsize=plot.big_figsize)

    # Plot the footprint
    area.plot(ax=ax, facecolor="white")

    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    # Plot all POIs
    pois.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
    plt.tight_layout()

def get_pois_and_df_map(town_city, df, tags, price_bin):
  """View of place of interest with scatter plot of house prices for a given town city"""
  df2 = df.loc[df.town_city==town_city].reset_index(drop=True)
  df2['price'] = np.digitize(df2['price'], price_bin)

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

  ax.set_xlim([west, east])
  ax.set_ylim([south, north])
  ax.set_xlabel("longitude")
  ax.set_ylabel("latitude")

  # Plot all POIs
  pois.plot(ax=ax, color="blue", alpha=0.7, markersize=10)

  colors = cm.rainbow(np.linspace(0, 1, 10))
  for price_bucket in range(1,11):
    df_1 = df2[df2['price'] == price_bucket]
    df_1 = gpd.GeoDataFrame(df_1, geometry=gpd.points_from_xy(df_1['longitude'], df_1['latitude']))
    df_1.plot(ax=ax, color=colors[price_bucket - 1], alpha=1, markersize=75)

  plt.tight_layout()


def count_nearby_pois(latitude, longitude, poi_tag, bounds, pois):
  if poi_tag not in pois.columns:
    return 0
  pois = pois[pois[key].notnull()]
  pois = pois[(pois['longitude'] < float(longitude) + bounds) & (pois['longitude'] > float(longitude) - bounds)]
  pois = pois[(pois['latitude'] < float(latitude) + bounds) & (pois['latitude'] > float(latitude) - bounds)]
  return len(pois)


def get_pois(place_name, df, poi_tags):
  """Given a place_name and tags, for each poi_tag find the number of pois in the vicinity of each house in df"""
  df = df.loc[df.town_city==place_name].reset_index(drop=True)
  north, south, east, west = get_lat_and_long_box(df)
  pois = ox.geometries_from_bbox(north, south, east, west, poi_tags)
  pois = pois.xs('node')
  pois[['longitude', 'latitude']] = pois.apply(lambda row : row.geometry.representative_point().coords[:][0], axis = 1).tolist()

  for poi_tag in poi_tags:
    df[poi_tag] = df.apply(lambda row : count_nearby_pois(row['latitude'], row['longitude'], key, 0.02, pois), axis = 1)
  return df


def scale_and_reduce(df, cols):
  """For PCA, some columns need to be scaled"""
  df_1 = df[cols]
  scaled = preprocessing.scale(df_1)
  df_1_s = pd.DataFrame(scaled, columns=cols)
  df = df.drop(cols, axis=1)
  return df.join(df_1_s)


def view_price(df):
  col = None
  while col not in df.columns:
    # Using this loop since I don't really want to do too much error handling
    print(df.columns)
    col = input("Enter the column name you want to view price against")
  plt.figure()
  plt.scatter(df[col], df['price'])
  plt.show()


def view_pca(df):
  dim = -1
  while dim not in {1,2,3}:
    # Using this loop since I don't really want to do too much error handling
    dim = input("Select a number 1 to 3 which corresponds to the number of dimensions you want to see the data in")
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

  elif dim == 3:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for _, row in df.iterrows():
      ax.scatter(row[0], row[1], row[2], color=colors[int(row['price']) - 1])
    plt.show()


def view_map(df):
  town_city_set = set(df.town_city.unique())
  town_city = None
  print(town_city_set)
  while town_city not in town_city_set:
    town_city = input("Which town city map do you want to view?")
  possible_tags = {"amenity": 0,"buildings": 0, "historic": 0, "leisure": 0, "shop": 0, "tourism": 0}
  for tag in possible_tags:
    value = bool(input(f"View tag {tag}? 1 for yes, 0 for no"))
    possible_tags[tag] = value
  price_bin = bin_price(df)
  get_pois_and_df_map(town_city, df, possible_tags, price_bin)


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    df.fillna(0)
    return df


def query(data):

    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data) -> None:
  """Provide a view of the data that allows the user to verify some aspect of its quality."""
  view_type = -1
  while view_type not in {0, 1, 2}:
    # Using this loop since I don't really want to do too much error handling
    view_type = input("""Enter 0 to view price with respect to a specific column in df, 1 to view price as as PCA scatter plot
                         2 to view a scatter plot map of house prices in a given town""")
  f = {0 : view_price, 1 : view_pca, 2 : view_map}
  f[view_type](data)


def labelled(data, town_city):
  """Provide a labelled set of data ready for supervised learning."""
  # Feature that I want to include unfortunately have to be hard coded in
  required_cols = ['price', 'date_of_transfer', 'postcode', 'latitude', 'longitude']
  one_hot_cols = ['pt_D', 'pt_F', 'pt_O', 'pt_S', 'pt_T']
  house_stats_cols = ['sold_before', 'sold_total', 'average_price_of_area']
  poi_tags = {'amenity' : True, 'shop' : True, 'tourism' : True, 'leisure' : True}
  df = get_pois(town_city, data, poi_tags)
  inverse_poi_tags = [f"inv_{tag}" for tag in poi_tags]
  for i_key, key in zip(inverse_poi_tags, poi_tags):
    df[i_key] = 1 / (1 + df[key])
  df = get_statistics_of_houses_sold_before(df)
  df = one_hot(df, 'pt', 'property_type')
  columns = required_cols + one_hot_cols + house_stats_cols + list(poi_tags.keys()) + inverse_poi_tags
  return df[columns]
