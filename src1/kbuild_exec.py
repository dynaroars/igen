import os.path
import subprocess as sp

def vcmd(cmd, inp=None, shell=True) -> tuple:
    proc = sp.Popen(cmd, shell=shell, stdin=sp.PIPE,
                    stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = proc.communicate(input=inp)
    return out.decode('utf-8'), err.decode('utf-8'), proc.returncode

def is_valid(options:dict) -> bool:
    contents = [] 
    for k,v in options.items():
        if v is None:
            s = "# {} is not set".format(k)
        else:
            s = "{}={}".format(k, v)
        contents.append(s)
    contents = '\n'.join(contents)
    
    with open('.config', 'w') as fh:
        fh.write(contents)

    cmd = 'make clean; scripts/kconfig/conf --silentoldconfig Kconfig'
    print(cmd)
    out, err, returncode = vcmd(cmd)
    out = out.strip()
    err = err.strip()
    print('out')
    print(out)
    print('err')
    print(err)
    print(type(returncode), returncode == 0)
    return not err and returncode == 0 



options = {
    'CONFIG_A':None,
    'CONFIG_B':'y',
    'CONFIG_C':'y',
    'CONFIG_D':'y',
    'CONFIG_E':'y',
    'CONFIG_F':'y',
    'CONFIG_G':'y'
    }

print(is_valid(options))