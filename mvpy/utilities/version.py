'''
Exposes version methods.
'''

import warnings
import operator
import packaging

def compare(v_a: str, op: str, v_b: str) -> bool:
    """Compare two version strings.
    
    Parameters
    ----------
    v_a : str
        Version a.
    op : str
        Comparison operator (<, <=, ==, !=, >=, >).
    v_b : str
        Version b
    
    Returns
    -------
    result : bool
        Result of the version comparison.
    """
    
    # define ops
    ops = {
        "<": "lt",
        "<=": "le",
        "==": "eq",
        "!=": "ne",
        ">=": "ge",
        ">": "gt"
    }
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        v_a = packaging.version.parse(v_a)
        v_b = packaging.version.parse(v_b)
        
        return getattr(operator, ops[op])(v_a, v_b)