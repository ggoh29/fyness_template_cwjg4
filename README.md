# Fynesse Template

This repo provides a python template repo for doing data analysis according to the Fynesse framework.

Each aspect is broken down into different subsections that make it easier to understand the process

## Access

The functions in the access file are broken down according to these categories

- Functions that are tasked with creating a connection to the database. These include creating credential files and attempting to connect to a pymysql database

- Functions for accessing specific queries about the data from the database. These functions usually require a connection object and thus are hidden from the user. Examples of functions in this category are functions that return dataframes filtered by year or by address.
 
- Frontend functions that users are intended to use to access data from the database. Functions in this category will usually call the functions listed above that deal with specific query as well as further opertaions such as joins or cleaning the data.

## Assess

The functions in the assess file are broken down according to these categories

- Functions that help prepare the data for other use cases such as providing binning or one hot encoding functionality.

- Functions that answer specific queries regarding the data.

- Functions that assist with viewing the dataframe. Currently, there are 3 different viewing options implemented. 1) Given a dataframe that has price as a column, view any other column as a 2d graph with respect to price. 2) Given a dataframe that has a longitude, latitude and price column, draw a 2d map plot of the houses in the dataframe as a street view where the colour of the house in the plot corresponds to its price. 3) Perform dimensionality reduction on a dataframe and view the result in 1d, 2d or 3d. 

- Functions that add extra features to a dataframe. Used in conjunction with assess.labelled and for supervised learning.
## Address

