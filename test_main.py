import unittest
from app import app

class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_predict(self):
        input_data = {
            'title': 'SNK',
            'genre[]': ['Adventure', 'Action'],
            'description': 'Titan',
            'type': 'OVA',
            'producer[]': ['ToysFactory', '3xCube'],
            'studio[]': ['33Collective', 'BrainsBase']
        }
        
        response = requests.post("http://localhost:5000/", input_data)

        result = response.json()

        assert self.assertIsInstance(result, float) and self.assertGreaterEqual(result, 0) and self.assertLessEqual(result, 10), f"The result is not what was expected"
        print("Test passed!")
     
    def stress_test(self):
        start_time = time.monotonic()
        for i in range(10):
            with self.subTest(i=i):
                self.test_predict()
        end_time = time.monotonic()
        execution_time = end_time - start_time
        self.assertLess(execution_time, 60, "It took to much time (over the 60s).")

if __name__ == '__main__':
    unittest.main()
