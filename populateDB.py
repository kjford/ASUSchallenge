'''
Load data from ASUS Kaggle Challege into MySQL database
data available at: http://www.kaggle.com/c/pakdd-cup-2014
To use:
Must have table creation privileges on a mysql server
change authent dictionary below to work with your server
'''

import pandas as pd
import MySQLdb as mdb


authent = {'host':'localhost',
           'user':'pyuser',
           'passwd':'testpass',
           'db':'asuschal'}

salefile = 'SaleTrain.csv'
repairfile = 'RepairTrain.csv'


print('Loading data from .csv files')
# load data from .csv file
sale = pd.read_csv(salefile)
repair = pd.read_csv(repairfile)

# define a function to make the date SQL appropriate
def sqldate(yrmo):
    yr,mo=yrmo.strip().split('/')
    return '%s-%s-01'%(yr,mo)

print('Opening MySQL connetions')
# set up connection to MySQL db asuschal
con = mdb.connect(**authent)

with con:
    cur = con.cursor()

    # create sales and repairs table:
    cur.execute("DROP TABLE IF EXISTS sales")
    cur.execute("""CREATE TABLE sales(
                 sale_id INT PRIMARY KEY AUTO_INCREMENT, 
                 module VARCHAR(10),
                 component VARCHAR(10),
                 sale_date DATE,
                 sale_num INT)""")

    cur.execute("DROP TABLE IF EXISTS repairs")
    cur.execute("""CREATE TABLE repairs(
                 repair_id INT PRIMARY KEY AUTO_INCREMENT,
                 module VARCHAR(10),
                 component VARCHAR(10),
                 sale_date DATE,
                 repair_date DATE,
                 repair_num INT)""")

    # add rows to db
    print('Adding data to db')
    for index, row in sale.iterrows():
        cur.execute("""INSERT INTO sales(
                    module,
                    component,
                    sale_date,
                    sale_num)
                    VALUES(
                    %s,%s,%s,%s)""",(row[0],row[1],sqldate(row[2]),row[3]))

    for index, row in repair.iterrows():
        cur.execute("""INSERT INTO repairs(
                    module,
                    component,
                    sale_date,
                    repair_date,
                    repair_num)
                    VALUES(
                    %s,%s,%s,%s,%s)""",(row[0],row[1],sqldate(row[2]),sqldate(row[3]),row[4]))

print('Done. Connection closed')