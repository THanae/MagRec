import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 22})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
matplotlib.rcParams['text.latex.unicode'] = True


def tex_escape(name: str):
    if '_' in name:
        spot = name.index('_')
        normal = name[:spot]
        subscript = name[spot+1:]
        name = r'${}_{{{}}}$'.format(normal, subscript) + '\n' + r'$ ({})$'.format(get_units(name))
        return name
    else:
        return r'${}$'.format(name) + '\n' + r'$ ({})$'.format(get_units(name))


def get_units(name):
    if name[0] == 'B' or name[0] == 'b':
        unit = 'nT'
    elif name[0] == 'v':
        unit = 'km s^{-1}'
    elif name[0] == 'n':
        unit = 'cm^{-3}'
    elif name[0] == 'T':
        unit = 'K'
    else:
        print(name[0])
        unit = ''
    return unit
