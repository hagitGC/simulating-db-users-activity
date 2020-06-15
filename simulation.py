import numpy as np
from collections import defaultdict
import pandas as pd

'''
class simulation:
**  samples the parameters (k and theta) of their Gamma distribution N (num_of_users) users, 
    and for each users generates T (num_of_timeframes) samples from Gamma distributions with the users parameters. 
    each sample represents the users risk for each time frame.
**  capacity 
    if the capacity is fixed - each time frame will equal to c (capacity_perc) * N (num_of_users)
    if the capacity is not fixed the stimulation samples a capacity for each time frame. 
**  noise 
    if noise > 0 the simulation will noise to the first time frame data (this data will be passed to the sampling_strategy
    as part of the init. 
**  model warmup time
    the time the system just learns the users before applying anomaly detection 
**  smoothing factor 
    smoothing factor is used as the proportion for exponential smoothing to reduce the randomality and ensure trends in the user's data 
**  change probability 
    The probability of a user to be compromised. That is the probability of a change in the users Gamma distribution function parameters.
**  method 
    The method used for calculating the users priors:
    Avg - the prior is set by the average risk found in the past time window
    Max - the prior is set by the maximal risk found in the past time window

'''
class simulation:
    def __init__(self, num_of_users, num_of_timeframes,  strategies  = ['epsilon greedy'],fixed_capacity = True,capacity_perc= 10,
                 smoothing_factor = 0.05,noise = 0, method = 'avg', change_prob=0.01, model_warmup_time = 20):  #rndseed,
        self.strategies = strategies
        self.gaps = {}
        self.capacity = {}
        self.oracle_score = {}
        self.oracle_score2 = {}
        self.oracle_lists = {}
        self.smoothing_factor = smoothing_factor
        self.num_of_timeframes = num_of_timeframes
        self.capacity_perc = capacity_perc
        self.num_of_users = num_of_users
        self.fixed_capacity = fixed_capacity
        self.noise = noise
        self.method = method
        self.change_prob = change_prob
        self.smoothing_factor = smoothing_factor
        self.modelwarmup = model_warmup_time
        self.changes_in_users = {} ## used to detect true / false positives for the anomaly detection
        self.dataset = self.createDataSet()
        for tf in range(0, num_of_timeframes):
            self.capacity[tf] = self.getCapacity()
            tf_users_risks = self.current_timeframe_risks(tf)
            sorted_users_and_risks = self.sort_users_by_risk(tf_users_risks)
            self.oracle_lists[tf] = self.choose_top_users(sorted_users_and_risks, self.capacity[tf])
            self.oracle_score[tf] =self.calc_TF_potential_risk(sorted_users_and_risks, self.capacity[tf])
            self.oracle_score2[tf] = self.calc_tf_total_discovered_risk(self.oracle_lists[tf])
        #self.prior_knowledge  = self.current_timeframe_risks(0)
        #self.strategy_obj  = sampling_strategy(self.prior_knowledge, strategies[0])
        self.tf_found_risk = defaultdict(dict)
        self.tf_regret = defaultdict(dict)
        #self.run_experiment()

    '''
    Calculate the capacity 
    if the capacity is fixed - it is calculated by the percent and number of users.
    if the capacity is not fixed it is randomly sampled  
    
    '''
    def getCapacity(self):
        if self.num_of_users < 6:
            capacity = min(self.num_of_users, 2)
        elif self.fixed_capacity:
            capacity = int(round(float(self.num_of_users) * float(self.capacity_perc) / 100))
        else:
            if self.num_of_users > 5:
                capacity = np.random.randint(2, int(round(self.num_of_users * 0.5)))
            else:
                capacity = 3
        return capacity


    '''
    randomly samle a users k and theta gamma parameters
    '''
    def createUser(self):
        k = np.random.random()  # a shape parameter
        theta = np.random.random()  # scale parameter
        return k, theta

    """
    getTimeFrameRisk
    sample a risk from a Gamma distribution with k and theta as parameters.
    return the risk. 
    """

    def getTimeFrameRisk(self, k, theta):
        sampRisk = np.random.gamma(k, theta)
        return sampRisk



    '''
    creates data set with trends (using  exponential smoothing):
    with probability = self.change_prob the user will be compromised 
    if self.changeprob  = 0 there will not be events of changes in the users parameters. 
    '''
    def createDataSet(self):
        usersTansarctions = []
        ## document the data to file named "chnges_in_users.txt":
        changes_in_users_file = open('changes_in_users.txt', 'a')
        for i in range(0, self.num_of_users):
            userID = i
            userParameters = self.createUser()
            transactionsList = []
            p = 0
            Change_length = 0
            for j in range(0, self.num_of_timeframes):
                if j> self.modelwarmup:
                    if p == 0:
                        if np.random.random() < self.change_prob :
                            while True:
                                newk, newtheta = self.createUser()
                                diffk = abs(newk - userParameters[0])
                                difftheta = abs(newtheta - userParameters[1])
                                if   difftheta > 0.4 or diffk > 0.4:
                                    break
                            Change_length = np.random.randint(200, 1000)
                            p = 1
                            transactionsList.append(self.getTimeFrameRisk(newk, newtheta))
                            if i not in self.changes_in_users.keys():
                                self.changes_in_users[i] = []
                            self.changes_in_users[i].append([j, j+Change_length])
                            ### write the change to the file:
                            line = "change user "+ str(i) +" time frame " + str(j) + " original users parameters "+ str(userParameters[0]) + " " + str(userParameters[1]) +"new parameters "+ str([newk, newtheta]) + " for " + str(Change_length) + '\n'
                            changes_in_users_file.write(line)
                            ###
                        else:
                            transactionsList.append(self.getTimeFrameRisk(userParameters[0], userParameters[1]))
                    else:
                        if p <= Change_length:
                            transactionsList.append(self.getTimeFrameRisk(newk, newtheta))
                            p += 1
                            if p == Change_length:
                                p = 0
                                Change_length = 0
                else:
                    transactionsList.append(self.getTimeFrameRisk(userParameters[0], userParameters[1]))
            smoothed_data = self.exponential_smoothing(transactionsList)
            usersTansarctions.append([userID, userParameters, smoothed_data])
        changes_in_users_file.close()
        return usersTansarctions

    '''
    exponential_smoothing
    Exponential smoothing is a rule of thumb technique for smoothing time series data using the exponential window function
    we use it to create trends and reduce the randomality in the data. 
    the function receives a series of risks of a user and returns the series values after the exponential smoothing.
    '''
    def exponential_smoothing(self, series):
        result = [series[0]]  # first value is same as series
        for n in range(1, len(series)):
            result.append(self.smoothing_factor * series[n] + (1 - self.smoothing_factor) * result[n - 1])
        return result

    '''
    current_timeframe_risks
    receives the time-frame index 
    return a list of tuples : [user, risk]   
    '''
    def current_timeframe_risks(self, timeFrameIndex):  #, users_data, timeFrameIndex
        time_frame_users_risk=[]
        for user in self.dataset:
            time_frame_users_risk.append([user[0], user[2][timeFrameIndex]])
        return time_frame_users_risk



    '''
    sort_users_by_risk
    sort a list of tuples [user, risk] by the second object (risk)
    '''
    def sort_users_by_risk(self, users_risk_list):
        sorted_by_second = sorted(users_risk_list, key=lambda tup: tup[1],reverse = True)
        return  sorted_by_second

    '''
    expected sorted list
    '''
    def calc_TF_potential_risk(self, users_risk_list, capacity):
        risk = 0
        riskslist = self.sort_users_by_risk(users_risk_list)
        if len(users_risk_list)<= capacity:
            capacity = len(users_risk_list)
        for userind in range(0, capacity):
            risk += riskslist[userind][1]
        return risk

    '''
    choose_top_users assumes users_risk_list is sorted in descending order
    used to create the oracle lists
    '''
    def choose_top_users(self, users_risk_list, capacity):
        TF_users = []
        for i in range(0, capacity):
            TF_users.append(users_risk_list[i])
        return TF_users

    '''
    get_users_risks
    gets users list and the current time frame and return a list of tuples of the users and their risks. 
    '''
    def get_users_risks(self, users_list, tf):
        time_frame_users_risk = []
        for user in users_list:
            for item in self.dataset:
                if user == item[0]:
                    time_frame_users_risk.append( [user, item[2][tf]])
                    break
        return time_frame_users_risk

    '''
    calc_tf_total_discovered_risk
    used for the evaluation of methods
    receives a list of tuples of [user, risk]
    returns the sum of all the risks
    '''
    def calc_tf_total_discovered_risk(self, users_and_risk):
        score = 0
        for user in users_and_risk:
            score += user[1]
        return score


    def run_experiment(self):
        prior_knowledge  = self.current_timeframe_risks(0)
        for strategy in self.strategies:
            experiment = sampling_strategy(prior_knowledge, strategy=strategy, method = self.method)
            for tf in range(1, self.num_of_timeframes):
                users_to_sample = experiment.get_users_list(capacity=self.capacity[tf],tf = tf, strategy= strategy )
                sampled_users_risks = self.get_users_risks(users_to_sample, tf)
                experiment.update_users_history(sampled_users_risks, tf)
                self.tf_found_risk[strategy][tf] =  self.calc_tf_total_discovered_risk(sampled_users_risks)
                self.tf_regret[strategy][tf] = 1- (self.tf_found_risk[strategy][tf]/self.oracle_score[tf])
                print "Time:", tf, "users that were sampled: ", users_to_sample, "oracles list:", self.oracle_lists[tf]
                print "found risk: ", self.tf_found_risk[strategy][tf], "Oracle risk for time:"  , self.oracle_score[tf], "the regret: ", self.tf_regret[strategy][tf]

    '''
    calc_statistics
    used for the evaluation of methods
    calculate the mean regret and mean recall for each srtategy for all the timeframes. 
    '''
    def calc_statistics(self):
        mean_regret = {}
        mean_recall = {}
        for strategy in self.strategies:
            mean_regret[strategy] = np.mean(self.tf_regret[strategy].values())
            mean_recall[strategy] = np.mean(self.tf_found_risk[strategy].values())/np.mean(self.oracle_score.values())
            print strategy, "regret", np.mean(self.tf_regret[strategy].values()), "recall", np.mean(self.tf_found_risk[strategy].values())/np.mean(self.oracle_score.values())
        return mean_regret, mean_recall, self.method

    '''
    is_true_positive
    used to evaluate the recall of anomaly detection. 
    receives a users and a time stamp (time frame index) and compares if to the list of changes in users 
    if the time is within the period that the user's destribution was changed - return True (true positive)
    if not - return False (false positive). 
    '''
    def is_true_positive(self, user, tf):
        if user in self.changes_in_users.keys():
            for times in self.changes_in_users[user]:
                if tf>times[0] and tf<times[1]:
                    return True
        return False



class sampling_strategy:
    def __init__(self, users_list_and_Prior, strategy = "baseline", window_size = 3, method = 'avg'):
        self.users_list = self.extract_users_list(users_list_and_Prior)
        self.window_size = window_size
        self.users_list_and_Prior = users_list_and_Prior
        self.users_history = defaultdict(dict)
        self.update_users_history(users_list_and_Prior, 0)
        self.users_priors = {}
        self.method = method
        self.update_users_prior(0)
        self.baseline_users = []


    def extract_users_list(self, users_list_and_Prior):
        users_list = []
        for user in users_list_and_Prior:
            users_list.append(user[0])
        return users_list

    '''
    updata_users_history updates the risk sampled for each user by the timeframe
    '''
    def update_users_history(self, users_risks, timeframe):
        for user in users_risks:
            self.users_history[user[0]][timeframe] = user[1]

    '''
    update_users_prior 
    calculate the prior for sampling using the method (avg or max) and the window size
    if there are no samples for the window the prior will remain unchanged ?????                 
    '''
    def update_users_prior(self, timeframe):
        if timeframe < self.window_size:
            if timeframe == 0 :
                window_times = [0]
            else:
                window_times = range(0,timeframe)
        else:
            window_times  = range((timeframe - self.window_size) , timeframe)
        for user in self.users_history.keys():
            risks = []
            for t in window_times:
                if t in self.users_history[user].keys():
                    risks.append(float(self.users_history[user][t]))
            if len(risks) > 0:
                if self.method == 'avg':
                    self.users_priors[user] = np.mean(risks)
                if self.method == 'max':
                    self.users_priors[user] = np.max(risks)

    '''
    choose_top_k_users:
    k - num of users to return 
    method  : 'avg' or 'max'
    based on the priors and the strategy it returns the top k users  
    '''
    def choose_top_k_users(self, capacity):
        users_and_priors = []
        for user in self.users_priors.keys():
            users_and_priors.append([user, self.users_priors[user] ])
        sorted_by_second = sorted(users_and_priors, key=lambda tup: tup[1], reverse=True)
        choosen_users  = self.extract_users_list(sorted_by_second[0:capacity])
        return choosen_users
    #

    '''
    get_users_list()
    gets a capacity, the current time frame and the chosen strategy and returns the list of users to be sampled
    '''
    def get_users_list(self, capacity, tf, strategy = 'epsilon greedy', risky_proportion = 0.8):
        if strategy == 'epsilon greedy':
            self.update_users_prior(tf)
            users_to_sample =  self.epsilon_greedy_sampling(capacity, risky_proportion)
            print "eps"
        if strategy == 'baseline':
            if tf == 1:
                self.baseline_users.append(self.choose_top_k_users(capacity)) #self.select_baseline_list(capacity)
            base_users_to_sample = self.baseline_users
            print base_users_to_sample
            users_to_sample = base_users_to_sample[0]
            print"base"
        if strategy == 'gibbs by risk':
            users_to_sample = self.GibbsByRisk(capacity)
            print "gibbs"
        return users_to_sample

    '''
    uses choose top k users. 
    assamble the list the following way: 
            (i)  risky proportion of the capacity is taken from the top of the users by their priors
            (ii) the reminder is sampled randomly from the entire list of users.  
    '''
    def epsilon_greedy_sampling(self,capacity, risky_proportion = 0.8 ):
        # MCMC sampling by risk
        TF_users = []
        #calculate the actual number of users in the capacity of the risky group:
        num_of_risky_users = int(round(float(capacity) * risky_proportion))
        #the reminder is the number of users that would be randomly selected
        remainder = int(capacity) - num_of_risky_users
        for i in range(0, num_of_risky_users):
            TF_users = self.choose_top_k_users(num_of_risky_users)
        ## sample from non risky users:
        if remainder > 0:
            i = 0
            while i<remainder:
                index = np.random.randint(0, len(self.users_list))
                if self.users_list[index] not in TF_users:
                    TF_users.append(self.users_list[index])
                    i+=1
        return TF_users

    def select_baseline_list(self, capacity):
        users_list = self.choose_top_k_users(capacity)
        return users_list

    '''
    ----------------------------------------------------------------------------------
    calc_probabilities_for_gibs(users_risk_list)
    receives a list of users and their known risks (the knowledge we own) and epsilon  
    returns - a list of these users and their gibbs probability 
              the max value (to be used by the GibbsByRisk algorithm)
    applied method: 
        1. sort the list by risk 
        2. for each user - assign probability as the sum of the accumulated risk and the users risk
        3. if the user is unknown (risk = 0 or null) use epsilon as his known risk.    
    ----------------------------------------------------------------------------------
    '''

    def calc_probabilities_for_gibs(self, epsilon=0.001):
        users_and_priors = []
        #convert form dict to np.array:
        for user in self.users_priors.keys():
            users_and_priors.append([user, self.users_priors[user]])
        #sort the list by the second object in each tuple (risk)
        sorted_risks = sorted(users_and_priors, key=lambda tup: tup[1], reverse=True)
        users_Gibbs_prob = []
        accum_prob = 0
        for user in sorted_risks:
            if user[1] > 0:
                accum_prob += user[1]
                # print user,  [user[0],accum_prob]
            else:
                print "im adding epsilon"
                accum_prob += epsilon
            users_Gibbs_prob.append([user[0], accum_prob])
        max_prob = accum_prob
        return users_Gibbs_prob, max_prob

    '''
    ----------------------------------------------------------------------------------
    GibbsByRisk(users_risk_list, capacity)
    receives a list of users and their known risks (the knowledge we own) and the sampling capacity 
    returns - the strategy : a list of  users that should be sampled next
    applied method: 
        1. assign probability for each user 
        2. dray users using the probability of choosing each one
        3. if the user is not in the list - add him /her to the list
    ----------------------------------------------------------------------------------
    '''

    def GibbsByRisk(self, capacity, epsilon=0.001):
        Gibbs_risks, max_prob = self.calc_probabilities_for_gibs(epsilon)
        choosen_users = []
        i = 0
        while i < capacity:
            prob = np.random.uniform(0, max_prob)
            sampled_user1 = (list(filter(lambda user: prob <= user[1], Gibbs_risks)))
            sampled_user = min(sampled_user1, key=lambda t: t[1])
            if (sampled_user[0] not in [x for x in choosen_users]):
                choosen_users.append(sampled_user[0])
                i += 1
        return choosen_users

if __name__ == '__main__':
    np.random.seed(4)
    simulation_obj = simulation(num_of_users=10, num_of_timeframes=100, strategies=['gibbs by risk','baseline', 'epsilon greedy'], fixed_capacity=True, capacity_perc=30, method='avg')
    simulation_obj.run_experiment()
    print "****-----------------------****"
    print "evaluate the experiment:"
    simulation_obj.calc_statistics()



