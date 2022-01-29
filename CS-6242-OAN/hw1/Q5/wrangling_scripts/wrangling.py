"""
cse6242 s21
wrangling.py - utilities to supply data to the templates.

This file contains a pair of functions for retrieving and manipulating data
that will be supplied to the template for generating the table. """
import csv

def username():
    return 'mpearl3'

def data_wrangling():
    with open('data/movies.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        table = list()
        # Feel free to add any additional variables
        
        # Read in the header
        for header in reader:
            break
        
        # Read in each row
        count = 1
        for row in reader:
            table.append(row)
            
            # Only read first 100 data rows - [2 points] Q5.a
            if count == 100:
                break
            else:
                count+=1

        # Order table by the last column - [3 points] Q5.b
        table.sort(key = lambda x: float(x[2]),reverse=True)
    
    return header, table

