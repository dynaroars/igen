#Temp functions
motiv_d = {(0, 0, 0, 0, 0, 0): ['L3', 'L4', 'L5'],
           (0, 0, 0, 0, 0, 1): ['L3', 'L4', 'L5'],
           (0, 0, 0, 0, 1, 0): ['L3', 'L4', 'L5', 'L6', 'L8'],
           (0, 0, 0, 0, 1, 1): ['L3', 'L4', 'L5', 'L6', 'L7'],
           (0, 0, 0, 1, 0, 0): ['L3', 'L4', 'L9'],
           (0, 0, 0, 1, 0, 1): ['L3', 'L4', 'L9'],
           (0, 0, 0, 1, 1, 0): ['L3', 'L4', 'L9'],
           (0, 0, 0, 1, 1, 1): ['L3', 'L4', 'L9'],
           (0, 0, 1, 0, 0, 0): ['L3', 'L4', 'L9'],
           (0, 0, 1, 0, 0, 1): ['L3', 'L4', 'L9'],
           (0, 0, 1, 0, 1, 0): ['L3', 'L4', 'L9'],
           (0, 0, 1, 0, 1, 1): ['L3', 'L4', 'L9'],
           (0, 0, 1, 1, 0, 0): ['L3', 'L4', 'L9'],
           (0, 0, 1, 1, 0, 1): ['L3', 'L4', 'L9'],
           (0, 0, 1, 1, 1, 0): ['L3', 'L4', 'L9'],
           (0, 0, 1, 1, 1, 1): ['L3', 'L4', 'L9'],
           (0, 1, 0, 0, 0, 0): ['L3', 'L4', 'L5'],
           (0, 1, 0, 0, 0, 1): ['L3', 'L4', 'L5'],
           (0, 1, 0, 0, 1, 0): ['L3', 'L4', 'L5', 'L6', 'L8'],
           (0, 1, 0, 0, 1, 1): ['L3', 'L4', 'L5', 'L6', 'L7'],
           (0, 1, 0, 1, 0, 0): ['L3', 'L4', 'L9'],
           (0, 1, 0, 1, 0, 1): ['L3', 'L4', 'L9'],
           (0, 1, 0, 1, 1, 0): ['L3', 'L4', 'L9'],
           (0, 1, 0, 1, 1, 1): ['L3', 'L4', 'L9'],
           (0, 1, 1, 0, 0, 0): ['L3', 'L4', 'L9'],
           (0, 1, 1, 0, 0, 1): ['L3', 'L4', 'L9'],
           (0, 1, 1, 0, 1, 0): ['L3', 'L4', 'L9'],
           (0, 1, 1, 0, 1, 1): ['L3', 'L4', 'L9'],
           (0, 1, 1, 1, 0, 0): ['L3', 'L4', 'L9'],
           (0, 1, 1, 1, 0, 1): ['L3', 'L4', 'L9'],
           (0, 1, 1, 1, 1, 0): ['L3', 'L4', 'L9'],
           (0, 1, 1, 1, 1, 1): ['L3', 'L4', 'L9'],
           (1, 0, 0, 0, 0, 0): ['L2', 'L4', 'L5'],
           (1, 0, 0, 0, 0, 1): ['L2', 'L4', 'L5'],
           (1, 0, 0, 0, 1, 0): ['L2', 'L4', 'L5', 'L6', 'L8'],
           (1, 0, 0, 0, 1, 1): ['L2', 'L4', 'L5', 'L6', 'L7'],
           (1, 0, 0, 1, 0, 0): ['L2', 'L4', 'L9'],
           (1, 0, 0, 1, 0, 1): ['L2', 'L4', 'L9'],
           (1, 0, 0, 1, 1, 0): ['L2', 'L4', 'L9'],
           (1, 0, 0, 1, 1, 1): ['L2', 'L4', 'L9'],
           (1, 0, 1, 0, 0, 0): ['L2', 'L4', 'L9'],
           (1, 0, 1, 0, 0, 1): ['L2', 'L4', 'L9'],
           (1, 0, 1, 0, 1, 0): ['L2', 'L4', 'L9'],
           (1, 0, 1, 0, 1, 1): ['L2', 'L4', 'L9'],
           (1, 0, 1, 1, 0, 0): ['L2', 'L4', 'L9'],
           (1, 0, 1, 1, 0, 1): ['L2', 'L4', 'L9'],
           (1, 0, 1, 1, 1, 0): ['L2', 'L4', 'L9'],
           (1, 0, 1, 1, 1, 1): ['L2', 'L4', 'L9'],
           (1, 1, 0, 0, 0, 0): ['L1', 'L4', 'L5'],
           (1, 1, 0, 0, 0, 1): ['L1', 'L4', 'L5'],
           (1, 1, 0, 0, 1, 0): ['L1', 'L4', 'L5', 'L6', 'L8'],
           (1, 1, 0, 0, 1, 1): ['L1', 'L4', 'L5', 'L6', 'L7'],
           (1, 1, 0, 1, 0, 0): ['L1', 'L4', 'L9'],
           (1, 1, 0, 1, 0, 1): ['L1', 'L4', 'L9'],
           (1, 1, 0, 1, 1, 0): ['L1', 'L4', 'L9'],
           (1, 1, 0, 1, 1, 1): ['L1', 'L4', 'L9'],
           (1, 1, 1, 0, 0, 0): ['L1', 'L4', 'L9'],
           (1, 1, 1, 0, 0, 1): ['L1', 'L4', 'L9'],
           (1, 1, 1, 0, 1, 0): ['L1', 'L4', 'L9'],
           (1, 1, 1, 0, 1, 1): ['L1', 'L4', 'L9'],
           (1, 1, 1, 1, 0, 0): ['L1', 'L4', 'L9'],
           (1, 1, 1, 1, 0, 1): ['L1', 'L4', 'L9'],
           (1, 1, 1, 1, 1, 0): ['L1', 'L4', 'L9'],
           (1, 1, 1, 1, 1, 1): ['L1', 'L4', 'L9']}

def get_cov(config,args):
    varnames = args['varnames']
    
    config_d = dict(config)
    var_vals = tuple([int(config_d[vname]) for vname in varnames])
    return motiv_d[var_vals]

