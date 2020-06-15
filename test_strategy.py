from simulation import simulation, sampling_strategy
import numpy as np
import unittest

num_of_users = 1
num_of_timeframes = 3
seednum = 5
users_risk_list = [(0, 0.8), (1, 0.7), (2, 0.6), (3, 0.4), (4, 0.8), (5, 0.9)]
users_risk_list2 = [(0, 0.9), (1, 0.8), (2, 0.11), (3, 0.5399999999999999), (4, 0.4), (5,0.99)]


class TestSamplingStrategy(unittest.TestCase):

    def test_samplint_strategy_init(self):
        np.random.seed(seednum)
        strategy_obj = sampling_strategy(users_risk_list)
        #print str(strategy_obj.users_history.items())
        self.assertEquals("{0: 0.8, 1: 0.7, 2: 0.6, 3: 0.4, 4: 0.8, 5: 0.9}", str(strategy_obj.users_priors))
        self.assertEquals("[(0, {0: 0.8}), (1, {0: 0.7}), (2, {0: 0.6}), (3, {0: 0.4}), (4, {0: 0.8}), (5, {0: 0.9})]", str(strategy_obj.users_history.items()))

    def test_extract_users_list(self):
        np.random.seed(seednum)
        strategy_obj = sampling_strategy(users_risk_list)
        users_list = strategy_obj.extract_users_list(users_risk_list)
        #print users_list
        self.assertEquals("[0, 1, 2, 3, 4, 5]", str(users_list))

    def test_update_users_history(self):
        np.random.seed(seednum)
        strategy_obj = sampling_strategy(users_risk_list)
        users_risks_list_time1 = [(0, 0.9), (1, 0.8), (3, 0.6), (4, 0.5)]
        strategy_obj.update_users_history(users_risks_list_time1, 1)
        # print strategy_obj.users_history.items()
        users_history = strategy_obj.users_history.items()
        self.assertEquals("[(0, {0: 0.8, 1: 0.9}), (1, {0: 0.7, 1: 0.8}), (2, {0: 0.6}), (3, {0: 0.4, 1: 0.6}), (4, {0: 0.8, 1: 0.5}), (5, {0: 0.9})]", str(users_history))
        users_risks_list_time2 = [(2, 0), (3, 0.58), (5, 0.99)]
        strategy_obj.update_users_history(users_risks_list_time2, 2)
        users_history = strategy_obj.users_history.items()
        self.assertEquals("[(0, {0: 0.8, 1: 0.9}), (1, {0: 0.7, 1: 0.8}), (2, {0: 0.6, 2: 0}), (3, {0: 0.4, 1: 0.6, 2: 0.58}), (4, {0: 0.8, 1: 0.5}), (5, {0: 0.9, 2: 0.99})]", str(users_history))
        users_risks_list_time3 = [(0, 0.9), (4, 0.3), (2, 0.22), (3, 0.44), (5, 0.99)]
        strategy_obj.update_users_history(users_risks_list_time3, 3)
        users_history = strategy_obj.users_history.items()
        self.assertEquals("[(0, {0: 0.8, 1: 0.9, 3: 0.9}), (1, {0: 0.7, 1: 0.8}), (2, {0: 0.6, 2: 0, 3: 0.22}), (3, {0: 0.4, 1: 0.6, 2: 0.58, 3: 0.44}), (4, {0: 0.8, 1: 0.5, 3: 0.3}), (5, {0: 0.9, 2: 0.99, 3: 0.99})]", str(users_history))

    def test_update_users_prior(self):
        np.random.seed(seednum)
        strategy_obj = sampling_strategy(users_risk_list)
        users_risks_list_time1 = [(0, 0.9), (1, 0.8), (3, 0.6), (4, 0.5)]
        strategy_obj.update_users_history(users_risks_list_time1, 1)
        # print strategy_obj.users_history.items()
        users_history = strategy_obj.users_history.items()
        users_risks_list_time2 = [(2, 0), (3, 0.58), (5, 0.99)]
        strategy_obj.update_users_history(users_risks_list_time2, 2)
        users_risks_list_time3 = [(0, 0.9), (4, 0.3), (2, 0.22), (3, 0.44), (5, 0.99)]
        strategy_obj.update_users_history(users_risks_list_time3, 3)
        strategy_obj.update_users_prior(4)
        self.assertEquals("{0: 0.9, 1: 0.8, 2: 0.11, 3: 0.5399999999999999, 4: 0.4, 5: 0.99}", str(strategy_obj.users_priors))


    def test_epsilon_greedy_sampling(self):
        np.random.seed(seednum)
        strategy_obj = sampling_strategy(users_risk_list)
        list = strategy_obj.epsilon_greedy_sampling(4)
        self.assertEquals('[5, 0, 4, 3]', str(list))
        list = strategy_obj.epsilon_greedy_sampling(4)
        self.assertEquals("[5, 0, 4, 1]", str(list))
        strategy_obj = sampling_strategy(users_risk_list2)
        list = strategy_obj.epsilon_greedy_sampling(4)
        self.assertEquals("[5, 0, 1, 4]", str(list))
        np.random.seed(13)
        list = strategy_obj.epsilon_greedy_sampling(4)
        self.assertEquals("[5, 0, 1, 2]", str(list))
        list = strategy_obj.epsilon_greedy_sampling(4)
        self.assertEquals("[5, 0, 1, 2]", str(list))


    def test_select_baseline_list(self):
        np.random.seed(seednum)
        strategy_obj = sampling_strategy(users_risk_list)
        self.assertEquals("[5, 0, 4, 1]", str(strategy_obj.select_baseline_list(4)))
        self.assertEquals("[5, 0, 4, 1, 2]", str(strategy_obj.select_baseline_list(5)))



    def test_calc_probabilities_for_gibs(self):
        np.random.seed(seednum)
        strategy_obj = sampling_strategy(users_risk_list)
        gibs_probs = strategy_obj.calc_probabilities_for_gibs()
        # the zeros after the 1.7 and 3.8 it is a bug of floats in python . ignor that / fix these numbers when converting to java
        self.assertEquals("([[5, 0.9], [0, 1.7000000000000002], [4, 2.5], [1, 3.2], [2, 3.8000000000000003], [3, 4.2]], 4.2)", str(gibs_probs))

    def test_GibbsByRisk(self):
        np.random.seed(seednum)
        strategy_obj = sampling_strategy(users_risk_list)
        list = strategy_obj.GibbsByRisk(4)
        self.assertEquals("[0, 2, 5, 3]", str(list))
        list = strategy_obj.GibbsByRisk(4)
        self.assertEquals("[4, 1, 2, 0]" , str(list))


if __name__ == '__main__':
    unittest.main()
