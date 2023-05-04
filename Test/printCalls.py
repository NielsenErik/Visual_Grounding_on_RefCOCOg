import time
import datetime



class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    ORANGE = '\033[33m'
    RESULT =  '\033[94m'
    WARNING = '\033[93m'
    DEBUGGING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def now():
    return datetime.datetime.now().strftime("%H:%M:%S")

def error(buffer):
    print(now()+" ["+colors.BOLD+colors.FAIL+"ERROR"+colors.ENDC+"] " + buffer +"\n")
    
def warning(buffer):
    print(now()+" ["+colors.BOLD+colors.ORANGE+"WARNING"+colors.ENDC+"] " + buffer +"\n")

def debugging(buffer):
    print(now()+" ["+colors.BOLD+colors.DEBUGGING+"DEBUG"+colors.ENDC+"] " + buffer +"\n")
    
def info(buffer):
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"INFO"+colors.ENDC+"] " + buffer +"\n")