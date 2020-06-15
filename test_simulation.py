from simulation import simulation
import numpy as np
import unittest

num_of_users = 1
num_of_timeframes = 3
seednum = 5

class TestSamplingByRisk(unittest.TestCase):

    def test_createUser(self):
        np.random.seed(seednum)
        # create an instance of the class simulation
        simulation_obj = simulation(num_of_users, num_of_timeframes)
        np.random.seed(seednum)
        u = simulation_obj.createUser()
        self.assertEquals("(0.22199317108973948, 0.8707323061773764)", str(u))

    def test_getTimeFrameRisk(self):
        simulation_obj = simulation(num_of_users, num_of_timeframes)
        np.random.seed(seednum)
        u = simulation_obj.createUser()
        self.assertEquals("0.000717660709373", str(simulation_obj.getTimeFrameRisk(u[0],u[1])))

    def test_createDataSet(self):
        np.random.seed(seednum)
        # create an instance of the class simulation
        simulation_obj = simulation(num_of_users, num_of_timeframes, smoothing_factor=0.1, change_prob= 0)
        dataset = simulation_obj.dataset
        self.assertEquals(1, len(dataset))
        self.assertEquals("(0.22199317108973948, 0.8707323061773764)", str(dataset[0][1]))
        self.assertEquals("0.000717660709373", str(dataset[0][2][0]))
        np.random.seed(seednum)
        u = simulation_obj.createUser()
        risks = [simulation_obj.getTimeFrameRisk(u[0], u[1]) for i in [0,1,2]]
        self.assertEquals(simulation_obj.exponential_smoothing(risks), dataset[0][2])

    def test_current_timeframe_risks(self):
        np.random.seed(seednum)
        # create an instance of the class simulation
        simulation_obj = simulation(3, 3, smoothing_factor=0.1, change_prob=0)
        risks = simulation_obj.current_timeframe_risks(1)
        #compare the risks from time frame index = 1 to the expected risks:
        self.assertEquals([[0,0.004097132462795419], [1, 0.0012278863428629003], [2,0.09637744711724958]], risks)

    def test_sort_users_by_risk(self):
        np.random.seed(seednum)
        # create an instance of the class simulation
        simulation_obj = simulation(3, 3, smoothing_factor=0.1, change_prob=0)
        risks = simulation_obj.current_timeframe_risks(1)
        sorted_by_risks = simulation_obj.sort_users_by_risk(risks)
        #compare the risks from time frame index = 1 to the expected risks:
        self.assertEquals([[2,0.09637744711724958], [0,0.004097132462795419], [1, 0.0012278863428629003]], sorted_by_risks)

    def test_choose_top_users(self):
        np.random.seed(seednum)
        # create an instance of the class simulation
        simulation_obj = simulation(3, 3, smoothing_factor=0.1, change_prob=0)
        users_risk_list = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5), (5, 0.4)]
        capacity = 2
        users = simulation_obj.choose_top_users(users_risk_list, capacity)
        self.assertEquals("[(0, 0.9), (1, 0.8)]", str(users))


    def test_culc_TF_potential_risk(self):
        np.random.seed(seednum)
        # create an instance of the class simulation
        simulation_obj = simulation(3, 3, smoothing_factor=0.1, change_prob=0)
        users_risk_list = [(0, 0.9), (1, 0.8), (3, 0.6), (4, 0.5), (2, 0), (5, 0)]
        capacity = 4
        users_risk_list2 = [(0, 0.9), (1, 0.8), (4, 0), (2, 0), (3, 0), (5, 0.99)]
        capacity2 = 3
        users_risk_list3 = [(0, 0.9), (1, 0.8), (4, 0.3), (2, 0.22), (3, 0.44), (5, 0.99)]
        capacity3 = 6
        capacity4 = 5
        self.assertEquals("2.8", str( simulation_obj.calc_TF_potential_risk(users_risk_list, capacity)))
        self.assertEquals("2.69", str(simulation_obj.calc_TF_potential_risk(users_risk_list2, capacity2)))
        self.assertEquals("3.65", str(simulation_obj.calc_TF_potential_risk(users_risk_list3, capacity3)))
        self.assertEquals("3.43", str(simulation_obj.calc_TF_potential_risk(users_risk_list3, capacity4)))

    def test_calc_tf_total_discovered_risk(self):
        np.random.seed(seednum)
        # create an instance of the class simulation
        simulation_obj = simulation(3, 3, smoothing_factor=0.1, change_prob=0)
        users_risk_list = [(0, 0.9), (1, 0.8), (3, 0.6), (4, 0.5), (2, 0), (5, 0)]
        risk = simulation_obj.calc_tf_total_discovered_risk(users_risk_list)
        self.assertEquals("2.8", str(risk))
        users_risk_list3 = [(0, 0.9), (1, 0.8), (4, 0.3), (2, 0.22), (3, 0.44), (5, 0.99)]
        risk =simulation_obj.calc_tf_total_discovered_risk(users_risk_list3)
        self.assertEquals("3.65", str(risk))




if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSamplingByRisk)
    unittest.TextTestRunner(verbosity=2).run(suite)