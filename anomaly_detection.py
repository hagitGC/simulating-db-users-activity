from simulation import sampling_strategy, simulation
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

'''
class anomaly_detection 
learns the behaviour of the users and decides if a new sample is an anomaly or not. 
'''
class anomaly_detection:
    def __init__(self):
        self.users_history = defaultdict(dict)

    '''
    update_users_history
    this function collects the data learned about each user. 
    receives: the time stemp (timeframe) and the list of <user, risk>  
    '''
    def update_users_history(self, users_risks, timeframe):
        for user in users_risks:
            self.users_history[user[0]][timeframe] = user[1]

    '''
    calc_mean_and_std
    recieves a user 
    :returns the mean and std as found from his khown history 
    '''
    def calc_mean_and_std(self, user):
        if user in self.users_history.keys():
            user_dist_mean = np.mean(np.array(self.users_history[user].values()))
            user_dist_std = np.std(np.array(self.users_history[user].values()))
            return user_dist_mean, user_dist_std
        else:
            return None, None

    '''
    is_this_anomaly
    uses the function calc_mean_and_std to calculate the users characteristics 
    if the new sample is within the mean +- 3 * std it is not it is not an anomaly. 
    '''
    def is_this_anomaly(self,user_risk_tuple):
        mean, std = self.calc_mean_and_std(user_risk_tuple[0])
        #looking for users that have higher risk
        if mean is not  None:
            if (user_risk_tuple[1] > mean + 3*std) or (user_risk_tuple[1] < mean - 3*std):
                #print "is anomaly, mean, std", mean, std, "user_risk_tuple", user_risk_tuple
                return True
            return False
        return False      #"insufficient data "