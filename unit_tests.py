import unittest

from analysis import scores, tornado_occurence_proportion, tornado_injury_proportion, tornado_fatality_proportion

class circle_ci_tests(unittest.TestCase):
    def test_ml_accuracy(self):
        
        function_output = scores.mean()
        minimum_accuracy = 0.8
        self.assertTrue(function_output > minimum_accuracy)

        val1 = tornado_occurence_proportion
        val2 = 0.02825746935648621
        self.assertEqual(val1, val2)

        val1 = tornado_injury_proportion
        val2 = 0.3081155433287483
        self.assertEqual(val1, val2)

        val1 = tornado_fatality_proportion
        val2 = 0.06543138390272148
        self.assertEqual(val1, val2)

        

if __name__ == '__main__':
    unittest.main()
