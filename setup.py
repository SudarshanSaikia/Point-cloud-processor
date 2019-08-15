from setuptools import setup


setup(
    name='Logistic',
    version='1.0',
    
    #name of the python module(.py file)
    py_modules=['logs'],

    #install required pyhton packages
    install_requires=[
        'Click','pandas','sklearn','pdal',
    ],
    entry_points='''
        [console_scripts]

    #log is the command that calls the cli function from the file logs.py
        ground=logs:ground
        non_ground=logs:non_ground
        train=logs:train
    ''',




)
