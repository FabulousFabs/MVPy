'''
Exposes checks for environment variables.
'''

import os

from typing import Optional

def get_var(var: str, default: Optional[str] = None, flag: Optional[str] = None) -> str:
    """Grab `var` from environment, respecting defaults and flag.
    
    Parameters
    ----------
    var : str
        The environment variable to get.
    default : Optional[str], default=None
        If `var` is None, what default to return (if any)?
    flag : Optional[str], default=None
        If supplied, this will override `var` for this call.
    
    Returns
    -------
    value : str
        Value of environment variable (or flag or default).
    """
    
    # check flag override
    if flag is not None:
        return flag

    # get environment var
    val = os.getenv(var)
    
    # fall back to default
    if val is None:
        return default

    return val
    
def is_enabled(var: str, default: bool = False, flag: Optional[bool] = None) -> bool:
    """Check if `var` is enabled in environment variables.
    
    Parameters
    ----------
    var : str
        The environment variable to get.
    default : bool, default=False
        If variable is not set, return default.
    flag : Optional[Union[int, bool]], default=None
        If supplied, this will override `var` for this call.
    
    Returns
    -------
    is_enabled : bool
        True if `var` is set to true (or was overriden, or default).
    """
    
    # setup matches
    match = ["1", "true", "yes", "on"]
    
    # check default
    if default:
        default = match[0]
    
    # check flag
    if flag:
        flag = match[0]
    
    # get val
    val = get_var(var, default = default, flag = flag)
    
    # check val
    return val.strip().lower() in match