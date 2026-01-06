import unittest


class subtraction_checks(unittest.TestCase):
    def test_unit(self):

        function_output = 8 - 2
        expected_output = 6
        self.assertEqual(function_output, expected_output, f"Fail: expected {expected_output}, got {function_output}")

        function_output = -24 - 5)
        expected_output = -29
        self.assertEqual(function_output, expected_output, f"Fail: expected {expected_output}, got {function_output}")

        function_output = -6 - -7
        expected_output = 1
        self.assertEqual(function_output, expected_output, f"Fail: expected {expected_output}, got {function_output}")


if __name__ == '__main__':
    unittest.main()
