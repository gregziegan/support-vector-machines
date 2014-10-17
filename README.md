# SSVM Solver

Another grand project for EECS 440

## Installation

    python setup.py install
    
If for some reason, this doesn't work:

    virtualenv env
    source env/bin/activate
    pip install -r requirements.txt
    
## Execution:

    python src/svm.py <data_file_name.ext> <C-value>
    python src/ssvm.py <data_file_name.ext> <C-value>
    