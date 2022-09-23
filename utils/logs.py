from pprint import PrettyPrinter
from termcolor import colored


pp = PrettyPrinter(indent=2).pprint

def cpprint(*x, c=None):
    for k in x:
        if isinstance(k, str):
            print(k if c is None else colored(k, c))
        else:
            pp(k)