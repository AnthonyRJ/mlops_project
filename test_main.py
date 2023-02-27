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
            'producer[]': ['', ''],
            'studio[]': ['', '']
        }
        
        response = requests.post("http://localhost:5000/", )

        result = response.json()
        expected_result = {}

        assert result == expected_result, f"Expected {expected_result} but got {result}"
        print("Test passed!")

if __name__ == '__main__':
    unittest.main()
