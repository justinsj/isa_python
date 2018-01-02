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

target_names=['0','1','2','3','4','5','6','7','8','9',
        'upwards force','downwards force','rightwards force','leftwards force', # Start from 10
        'counter-clockwise moment', 'clockwise moment','unknown','random', # Start from 14
        'fixed support','pinned support','roller support vertical', 'roller support horizontal', # Start from 18
        'uniformly distributed load', 'linearly distributed load','quadratically distributed load', 'cubically distributed load', # Start from 22
        'horizontal beam','vertical beam','downward diagonal beam', 'upward diagonal beam', # Start from 26
        'length','height','counter-clockwise angle','clockwise angle', # Start from 30
        'measure left','measure right', 'measure up','measure down', # Start from 34
        ]



#subset_definitions maps categories to class indices (ex. forces are classes 10,11,12, and 13)
subset_dictionary = {'numbers':[0,1,2,3,4,5,6,7,8,9],
'forces' : [10,11,12,13],
'counter_clockwise_moments' : [14,15,16,17],
'clockwise_momets' : [18,19,20,21],
'moments' : [14,15,16,17,18,19,20,21],

'fixed_supports_vertical' : [26,27,30,31],
'fixed_supports_horizontal' : [24,25,28,29],
'fixed_supports' : [24,25,26,27,28,29,30,31],

'pinned_supports_vertical' : [32,33,36,37],
'pinned_supports_horizontal' : [34,35,38,39],
'pinned_supports' : [32,33,34,35,36,37,38,39],

'roller_supports_vertical' : [40,41,44,45],
'roller_supports_horizontal' : [42,43,46,47],
'roller_supports' : [40,41,42,43,44,45,46,47],

'distributed_loads' : [48,49,50,51],
'beams' : [52,53,54,55],
'dimensions' : [56,57,58,59],
'measures' : [60,61,62,63]
}
specific_dictionary = {
        '0':[0],'1':[1],'2':[2],'3':[3],'4':[4],'5':[5],'6':[6],'7':[7],'8':[8],'9':[9],
        'upwards force':[10],'downwards force':[11],'rightwards force':[12],'leftwards force':[13],  # Start from 10
        'counter-clockwise moment right':[14], 'counter-clockwise moment up':[15], 'counter-clockwise moment left':[16], 'counter-clockwise moment down':[17], # Start from 14
        'clockwise moment right':[18],'clockwise moment up':[19],'clockwise moment left':[20],'clockwise moment down':[21], # Start from 18
        'unknown':[22],'random alphabet':[23], # Start from 22
        'fixed support right':[24],'fixed support left':[25],'fixed support down':[26], 'fixed support up':[27], # Start from 24
        'fixed support right w/ beam':[28],'fixed support left w/ beam':[29],'fixed support down w/ beam':[30], 'fixed support up w/ beam':[31], # Start from 28
        'pinned support down':[32], 'pinned support up':[33],'pinned support left':[34], 'pinned support right':[35], # Start from 32
        'pinned support down w/ beam':[36], 'pinned support up w/ beam':[37],'pinned support left w/ beam':[38], 'pinned support right w/ beam':[39], # Start from 36
        'roller support down':[40], 'roller support up':[41],'roller support left':[42],'roller support right':[43], # Start from 40
        'roller support down w/ beam':[44], 'roller support up w/ beam':[45],'roller support left w/ beam':[46],'roller support right w/ beam':[47], # Start from 44
        'uniformly distributed load':[48], 'linearly distributed load':[49],'quadratically distributed load':[50], 'cubically distributed load':[51], # Start from 48
        'horizontal beam':[52],'vertical beam':[53],'downward diagonal beam':[54], 'upward diagonal beam':[55], # Start from 52
        'length':[56],'height':[57],'counter-clockwise angle':[58],'clockwise angle':[59], # Start from 56
        'measure left':[60],'measure right':[61],'measure up':[62],'measure down':[63] # Start from 60
        }