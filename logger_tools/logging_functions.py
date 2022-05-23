import logging
import json
import functools
import inspect
import sys

LOGGER = logging.getLogger(__name__)
handler = logging.FileHandler('outputs1.log', mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s -%(message)s')
handler.setFormatter(formatter)
LOGGER.addHandler(handler)

def write_log(function_name, level, class_attributes=None, method_varibles=None, error_message=None ):
    
    level_lookup = {
        'debug':10,
        'info':20,
        'warning':30,
        'error':40,
        'critical':50
    }

    json_message = {
        'function':function_name,
        'class_attributes':class_attributes,
        'method_varibles':method_varibles,
        'error_message':error_message
    }
    json_message = json.dumps(json_message)

    LOGGER.log(level=level_lookup[level],msg=json_message)



def logging_decorator(func):  
    
    @functools.wraps(func)
    def wrapper(*args,**kwargs):

        # Get list of all arguments in class call
        class_attr = __get_class_attributes(args[0])
        # Get list of all arguments going into function call
        args_repr = [repr(a) for a in args[1:]]          # Ignore first argument as this wil alwasy be the instance of the class
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        func_attr = ", ".join(args_repr + kwargs_repr)

        try:
            out = func(*args,**kwargs)
            write_log(func.__name__,'info',class_attributes=class_attr,method_varibles=func_attr )
            return out
        except Exception as e:
            write_log(func.__name__,level='error',class_attributes=class_attr, method_varibles=func_attr, error_message=repr(e))
            raise
    return wrapper
 
def __get_class_attributes(class_instance):
     # pulls attributes used to initalise class

    # get names of params used to initalise the class
     sig = inspect.signature(class_instance.__init__)
     param_list = list(sig.parameters.keys())
     init_dict = {}
     # match param names up with values used when initalising
     for param in param_list:
        # try all params of starting class, not all inherited classes will share these params
        try:
            init_dict[param] = class_instance.__dict__[param] # make this more efficient?
        except:
            continue
     return init_dict


