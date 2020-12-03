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

class DBconnect:
    # TODO (@jmhobbs29)
    def __init__(self,):
        """
        Initialization of class - defines what database to connect to and what tables to use
        Will update master table with metadata about current iteration of the model, and create a new table containing every job
            that will be processed in this iteration of the model, as well as some parameters that will be updated later.  As we will want
            this to take every job that will be run, we will need to initialize this in the database after jobs have been created and
        """
        print('Connecting to postgres database...')
        conn = psycopg2.connect(
            # update to ini file for enhance security
            host="localhost",
            database="postgres",
            user="postgres",
            password="postgres")

        cur = conn.cursor()
        cur.execute('SELECT version()')
        db_version = cur.fetchone()
        print(db_version)
        #cur.close()
        #pass

    def updateparams(values):
        conn = psycopg2.connect(
            # update to ini file for enhance security
            host="localhost",
            database="postgres",
            user="postgres",
            password="postgres")

        cur = conn.cursor()
        cur.execute('SELECT count(*) FROM skynet_params')
        count = cur.fetchone()
        print(count[0])
        if count[0] == 0:
            sql = 'INSERT INTO skynet_params (resources, syscap, long, short, resourcelow, resourcehigh, timelow, timehigh, jobs) values (%s, %s, %s, %s, %s, %s, %s, %s, %s)'
            cur.execute(sql, (values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]))
            conn.commit()
            print("Parameters have been instantiated.")
        else:
            sql = 'UPDATE skynet_params set resources = %s, syscap = %s, long = %s, short = %s, resourcelow = %s, resourcehigh = %s, timelow = %s, timehigh = %s, jobs =%s'
            cur.execute(sql, (values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]))
            conn.commit()
            print("Parameters have been updated.")

    def updatejob():
        """
        This function will update the jobs in the database each time the environment moves to a new stage, with info on the length it took and what
        timestep the job was completed in, etc.
        """
        pass


    def updatemaster():
        """
        This function will update the master table at the end of the iteration with aggregate statistics, including whatever statistic the
        model is being limited on (for instance, total time to completion or average job slowdown)
        """
        pass




if __name__ == '__main__':
   DBconnect()

