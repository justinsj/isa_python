""" All the constants """

# Correct indices
target_names_all = [
    '0','1','2','3','4','5','6','7','8','9',
    'upwards force','downwards force','rightwards force','leftwards force',  # Start from 10
    'counter-clockwise moment right', 'counter-clockwise moment up', 'counter-clockwise moment left', 'counter-clockwise moment down', # Start from 14
    'clockwise moment right','clockwise moment up','clockwise moment left','clockwise moment down', # Start from 18
    'unknown','random alphabet', # Start from 22
    'fixed support right','fixed support left','fixed support down', 'fixed support up', # Start from 24
    'fixed support right w/ beam','fixed support left w/ beam','fixed support down w/ beam', 'fixed support up w/ beam', # Start from 28
    'pinned support down', 'pinned support up','pinned support left', 'pinned support right', # Start from 32
    'pinned support down w/ beam', 'pinned support up w/ beam','pinned support left w/ beam', 'pinned support right w/ beam', # Start from 36
    'roller support down', 'roller support up','roller support left','roller support right', # Start from 40
    'roller support down w/ beam', 'roller support up w/ beam','roller support left w/ beam','roller support right w/ beam', # Start from 44
    'uniformly distributed load', 'linearly distributed load','quadratically distributed load', 'cubically distributed load', # Start from 48
    'horizontal beam','vertical beam','downward diagonal beam', 'upward diagonal beam', # Start from 52
    'length','height','counter-clockwise angle','clockwise angle', # Start from 56
    'measure left','measure right','measure up','measure down' # Start from 60
]

target_names=[
    '0','1','2','3','4','5','6','7','8','9',
    'upwards force','downwards force','rightwards force','leftwards force', # Start from 10
    'counter-clockwise moment', 'clockwise moment','unknown','random', # Start from 14
    'fixed support','pinned support','roller support vertical', 'roller support horizontal', # Start from 18
    'uniformly distributed load', 'linearly distributed load','quadratically distributed load', 'cubically distributed load', # Start from 22
    'horizontal beam','vertical beam','downward diagonal beam', 'upward diagonal beam', # Start from 26
    'length','height','counter-clockwise angle','clockwise angle', # Start from 30
    'measure left','measure right', 'measure up','measure down', # Start from 34
]

target_names_dict = [
    'number': ['0','1','2','3','4','5','6','7','8','9'],
    'force': ['upwards force','downwards force','rightwards force','leftwards force'],  # Start from 10
    'counter-clockwise-moment': ['counter-clockwise moment right', 'counter-clockwise moment up', 'counter-clockwise moment left', 'counter-clockwise moment down'], # Start from 14
    'clockwise-moment': ['clockwise moment right','clockwise moment up','clockwise moment left','clockwise moment down'], # Start from 18
    'random': ['unknown','random alphabet'], # Start from 22
    'fixed-support': ['fixed support right','fixed support left','fixed support down', 'fixed support up',
                        'fixed support right w/ beam','fixed support left w/ beam','fixed support down w/ beam', 'fixed support up w/ beam'], # Start from 24
    'pinned-support': ['pinned support down', 'pinned support up','pinned support left', 'pinned support right',
                        'pinned support down w/ beam', 'pinned support up w/ beam','pinned support left w/ beam', 'pinned support right w/ beam'], # Start from 32
    'roller-vertical': ['roller support down', 'roller support up', 'roller support down w/ beam', 'roller support up w/ beam'],
    'roller-horizontal': ['roller support left','roller support right', 'roller support left w/ beam','roller support right w/ beam'],
    'distributed-load': ['uniformly distributed load', 'linearly distributed load','quadratically distributed load', 'cubically distributed load'],
    'beam': ['horizontal beam','vertical beam','downward diagonal beam', 'upward diagonal beam'], 
    'dimension': ['length','height','counter-clockwise angle','clockwise angle'],
    'measure': ['measure left','measure right','measure up','measure down']
]