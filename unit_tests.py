import unittest

from analysis import scores

class circle_ci_tests(unittest.TestCase):
    def test_ml_accuracy(self):
        
        function_output = scores.mean()
        minimum_accuracy = 0.8
        self.assertTrue(function_output > minimum_accuracy)

        

if __name__ == '__main__':
    unittest.main()
