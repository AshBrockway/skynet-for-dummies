"""
Author: Andrew Reilly, andrew.reilly19@ncf.edu
Date: 10/10/20
Purpose: This file contains multiple classses, outlining each of the comparison methods from the DeepRM paper:
    1. Shortest Job First - allocate jobs based on the job duration
    2. Packer - Allocate jobs based on resource availability
    3. FIFO - Requested by Novetta, simply allocates jobs based on the order they arrive
    4. ‘Tetris’ - An pseudo-intelligent agent that creates a combined score between job duration and resource availability and allocates jobs
        based on that score.
        *Note - The method for this is not obvious from either paper, will need to look through the DeepRM repository to get a better idea how it works

    The idea for these classes is to function similar to the DPN - at each step, select what jobs to process, send it back to the enviornment

"""

class SJF():
    def __init__(self,):
        """
        initialization of this method (if needed)
        """
        pass

    def select_policies():
        """
        the function to select what jobs to send to enviornment
        """
        pass


class Packer():
    def __init__(self,):
        """
        initialization of this method (if needed)
        """
        pass

    def select_policies():
        """
        the function to select what jobs to send to enviornment
        """
        pass



class FIFO():
    def __init__(self,):
        """
        initialization of this method (if needed)
        """
        pass

    def select_policies():
        """
        the function to select what jobs to send to enviornment
        """
        pass



class Tetris():
    def __init__(self,):
        """
        initialization of this method
        """
        pass

    def select_policies():
        """
        the function to select jobs, based on analyze_job() function.  This is the output to send to enviornment
        """
        pass

    def analyze_job():
        """
        this function will analyze a spectific job from those that can be picked from
        """
        pass
