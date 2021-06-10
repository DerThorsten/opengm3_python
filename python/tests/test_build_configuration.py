import unittest
import opengm



class TestBuildConfig(unittest.TestCase):
    
    def test_version(self):
        config = opengm.BuildConfiguration
        self.assertEqual(config.VERSION_MAJOR, 0)
        self.assertEqual(config.VERSION_MINOR, 1)
        self.assertEqual(config.VERSION_PATCH, 0)

