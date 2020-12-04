"""
Author: Andrew Reilly, andrew.reilly19@ncf.edu, Jamie Hobbs, jamie.hobbs19@ncf.edu
=======

Date: 10/10/20
Purpose: The class in this file is designed to take jobs and parameters from our model and store them in a database for
use in frontend dashboard visualizing the network.

Database layout:
    The database will consist of one single master table, with an id for each iteration of the model as well as information about the
        overall parameters in the model (resource capacity, number of jobs, type of model, etc.)
    The database will also consist of a table for each iteration of the model, named after the id found in the master table for easy linking,
        containing every job the iteration will run and statistics about it (resources needed, dominant resource, time to complete, actual slowdown)

Using a schema as the one described above will allow our dashboard to have two stages - a (potentially live) visualization of the model as it is running,
and a comparison view showing comparisons of models (over time)
"""

import psycopg2

def get_last_iter():
    conn = psycopg2.connect(
            # update to ini file for enhance security
            host="localhost",
            database="postgres",
            user="postgres",
            password="postgres")
    cur = conn.cursor()
    cur.execute('SELECT max(iteration) FROM skynet_params')
    last_iter = cur.fetchone()
    cur.close()
    return (last_iter[0])


class DBconnect:
    def __init__(self, cur_iter):
        """
        Initialization of class - defines what database to connect to and what tables to use
        Will update master table with metadata about current iteration of the model, and create a new table containing every job
            that will be processed in this iteration of the model, as well as some parameters that will be updated later.  As we will want
            this to take every job that will be run, we will need to initialize this in the database after jobs have been created and
        """
        print('Connecting to postgres database...')
        self.conn = psycopg2.connect(
            # update to ini file for enhance security
            host="localhost",
            database="postgres",
            user="postgres",
            password="postgres")

        self.cur = self.conn.cursor()
        self.iter = cur_iter

        #creating iteration table if not already in there
        table = "iter"+str(cur_iter)
        print(table)
        sql = 'CREATE TABLE IF NOT EXISTS {} ("epoch" int, "DPN_loss" numeric, "DPCNN_loss" numeric, "AC_loss" numeric, "FIFO_loss" numeric, "SJF_loss" numeric, "Random_loss" numeric, "Packer_loss" numeric, "Tetris_loss" numeric);'.format(table)
        self.cur.execute(sql)
        self.conn.commit()


        #checking db version
        self.cur.execute('SELECT version()')
        db_version = self.cur.fetchone()
        print('database version', db_version)
        self

    #VALUES must be in list form: [n_resources, syscap, %long, %short, resourcelow, resourcehigh, timelow, timehigh, n_jobs, n_epochs]
    def updateparams(self, values):
        sql = 'SELECT count(*) FROM skynet_params WHERE iteration = %s;'
        self.cur.execute(sql, (self.iter))
        count = self.cur.fetchone()
        #print(count[0])
        if count[0] == 0:
            sql = 'INSERT INTO skynet_params (iteration, resources, syscap, long, short, resourcelow, resourcehigh, timelow, timehigh, jobs, DPN_loss, DPCNN_loss, AC_loss, FIFO_loss, SJF_loss, Random_loss, Packer_loss, Tetris_loss) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0, 0, 0, 0, 0, 0, 0, 0);'
            self.cur.execute(sql, (self.iter, values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]))
            self.conn.commit()
            print("Parameters have been instantiated.")
        else:
            sql = 'UPDATE skynet_params SET resources = %s, syscap = %s, long = %s, short = %s, resourcelow = %s, resourcehigh = %s, timelow = %s, timehigh = %s, jobs =%s WHERE iteration = %s;'
            self.cur.execute(sql, (values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8], self.iter))
            self.conn.commit()
            print("Parameters have been updated.")

    #OVERALL LOSSES must be in list form: [DPN_loss, DPCNN_loss, AC_loss, FIFO_loss, SJF_loss, Random_loss, Packer_loss, Tetris_loss]
    def update_master(self, losses):
        sql = 'UPDATE skynet_params SET DPN_loss = %s, DPCNN_loss = %s, AC_loss = %s, FIFO_loss = %s, SJF_loss = %s, Random_loss = %s, Packer_loss = %s, Tetris_loss = %s WHERE iteration = %s;'
        self.cur.execute(sql, (losses[0], losses[1], losses[2], losses[3], losses[4], losses[5], losses[6], losses[7], self.iter))
        self.conn.commit()
        print("Overall Losses have been updated.")

    #INTERMEDIATE LOSSES must be in list form: [DPN_loss, DPCNN_loss, AC_loss, FIFO_loss, SJF_loss, Random_loss, Packer_loss, Tetris_loss]
    def updateiteration(self, losses, cur_epoch):
        sql = 'INSERT INTO %s SET epoch = %s, DPN_loss = %s, DPCNN_loss = %s, AC_loss = %s, FIFO_loss = %s, SJF_loss = %s, Random_loss = %s, Packer_loss = %s, Tetris_loss = %s;'
        self.cur.execute(sql, (self.iter, cur_epoch, losses[0], losses[1], losses[2], losses[3], losses[4], losses[5], losses[6], losses[7]))
        self.conn.commit()
        print("Intermediate Losses have been updated.")

    def close_conn(self):
        self.cur.close()


#testing
#get_last_iter()

db = DBconnect(cur_iter=12)
fake_params = [4, 3, 30, 70, .1, .5, 2, 15, 70, 50]
fake_params[0]
db.updateparams(fake_params)




