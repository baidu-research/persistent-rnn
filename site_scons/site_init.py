
EnsureSConsVersion(1,2)

import os
import sys

import inspect
import platform
import re
import subprocess
from SCons import SConf

def getTools():
    result = []
    if os.name == 'nt':
        result = ['nvcc', 'default', 'msvc']
    elif os.name == 'posix':
        result = [ 'nvcc', 'default','g++']
    else:
        result = [ 'nvcc', 'default']

    return result;


OldEnvironment = Environment;

# this dictionary maps the name of a compiler program to a dictionary mapping the name of
# a compiler switch of interest to the specific switch implementing the feature
gCompilerOptions = {
        'gcc' : {'warn_all' : '-Wall',
            'warn_errors' : '-Werror',
            'optimization' : '-O3', 'debug' : '-g',
            'exception_handling' : '', 'standard': ''},
        'clang' : {'warn_all' : '-Wall',
            'warn_errors' : '-Werror',
            'optimization' : '-O3', 'debug' : '-g',
            'exception_handling' : '', 'standard': ''},
        'g++' : {'warn_all' : '-Wall',
            'warn_errors' : '-Werror',
            'optimization' : '-O3', 'debug' : '-g',
            'exception_handling' : '', 'standard': '-std=c++11'},
        'c++' : {'warn_all' : '-Wall',
            'warn_errors' : '-Werror',
            'optimization' : '-O3', 'debug' : '-g',
            'exception_handling' : '',
            'standard': ['-stdlib=libc++', '-std=c++0x', '-pthread']},
        'clang++' : {'warn_all' : '-Wall',
            'warn_errors' : '-Werror',
            'optimization' : ['-O3'], 'debug' : ['-g'],
            'exception_handling' : '',
            'standard': ['-stdlib=libc++', '-std=c++11', '-pthread']},
        'cl'  : {'warn_all' : '/Wall',
                 'warn_errors' : '/WX',
                 'optimization' : ['/Ox', '/MD', '/Zi', '/DNDEBUG'],
                 'debug' : ['/Zi', '/Od', '/D_DEBUG', '/RTC1', '/MDd'],
                 'exception_handling': '/EHsc',
                 'standard': ['/GS', '/GR', '/Gd', '/fp:precise',
                     '/Zc:wchar_t','/Zc:forScope', '/DYY_NO_UNISTD_H']}
    }


# this dictionary maps the name of a linker program to a dictionary mapping the name of
# a linker switch of interest to the specific switch implementing the feature
gLinkerOptions = {
        'gcc'  : {'debug' : '', 'libraries' : ''},
        'clang'  : {'debug' : '', 'libraries' : ''},
        'g++'  : {'debug' : '', 'libraries' : ''},
        'c++'  : {'debug' : '', 'libraries' : '-lc++'},
        'clang++'  : {'debug' : '', 'libraries' : '-lc++'},
        'link' : {'debug' : '/debug', 'libraries' : ''}
    }

def getCFLAGS(mode, warn, warnings_as_errors, CC):
    result = []
    if mode == 'release':
        # turn on optimization
        result.append(gCompilerOptions[CC]['optimization'])
    elif mode == 'debug':
        # turn on debug mode
        result.append(gCompilerOptions[CC]['debug'])
        result.append('-DPRNN_DEBUG')

    if warn:
        # turn on all warnings
        result.append(gCompilerOptions[CC]['warn_all'])

    if warnings_as_errors:
        # treat warnings as errors
        result.append(gCompilerOptions[CC]['warn_errors'])

    result.append(gCompilerOptions[CC]['standard'])

    return result

def getLibCXXPaths():
    """Determines libc++ path

    returns (inc_path, lib_path)
    """

    # determine defaults
    if os.name == 'posix':
        inc_path = '/usr/include'
        lib_path = '/usr/lib/libc++.so'
    else:
        raise ValueError, 'Error: unknown OS.  Where is libc++ installed?'

    # override with environement variables
    if 'LIBCXX_INC_PATH' in os.environ:
        inc_path = os.path.abspath(os.environ['LIBCXX_INC_PATH'])
    if 'LIBCXX_LIB_PATH' in os.environ:
        lib_path = os.path.abspath(os.environ['LIBCXX_LIB_PATH'])

    return (inc_path, lib_path)

def getCXXFLAGS(mode, warn, warnings_as_errors, CXX):
    result = []
    if mode == 'release':
        # turn on optimization
        result.append(gCompilerOptions[CXX]['optimization'])
    elif mode == 'debug':
        # turn on debug mode
        result.append(gCompilerOptions[CXX]['debug'])
    # enable exception handling
    result.append(gCompilerOptions[CXX]['exception_handling'])

    if warn:
        # turn on all warnings
        result.append(gCompilerOptions[CXX]['warn_all'])

    if warnings_as_errors:
        # treat warnings as errors
        result.append(gCompilerOptions[CXX]['warn_errors'])

    result.append(gCompilerOptions[CXX]['standard'])

    return result

def getLINKFLAGS(mode, LINK):
    result = []
    if mode == 'debug':
        # turn on debug mode
        result.append(gLinkerOptions[LINK]['debug'])

    result.append(gLinkerOptions[LINK]['libraries'])

    return result

def cuda_exists(env):
    if not env['with_cuda']:
        return False
    return os.path.exists(env['cuda_path'])

def getExtraLibs(env):
    if os.name == 'nt':
        return []
    else:
        if cuda_exists(env):
            return ['cudart_static']
        else:
            return []

def importEnvironment():
    env = {  }

    if 'PATH' in os.environ:
        env['PATH'] = os.environ['PATH']

    if 'CXX' in os.environ:
        env['CXX'] = os.environ['CXX']

    if 'CC' in os.environ:
        env['CC'] = os.environ['CC']

    if 'TMP' in os.environ:
        env['TMP'] = os.environ['TMP']

    if 'LD_LIBRARY_PATH' in os.environ:
        env['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']

    return env

def updateEnvironment(env):
    originalEnvironment = importEnvironment()

    for key, value in originalEnvironment.iteritems():
        env[key] = value

def BuildEnvironment():
    vars = Variables()

    # add a variable to handle RELEASE/DEBUG mode
    vars.Add(EnumVariable('mode', 'Release versus debug mode', 'debug',
        allowed_values = ('release', 'debug')))

    # add a variable to handle warnings
    vars.Add(BoolVariable('Wall', 'Enable all compilation warnings', 1))

    # shared or static libraries
    libraryDefault = 'shared'

    vars.Add(EnumVariable('library', 'Build shared or static library',
        libraryDefault, allowed_values = ('shared', 'static')))

    # add a variable to treat warnings as errors
    vars.Add(BoolVariable('Werror', 'Treat warnings as errors', 1))

    # enable_cuda
    vars.Add(BoolVariable('with_cuda', 'Enable cuda', 1))

    # add a variable to determine the install path
    default_install_path = '/usr/local'

    if 'PRNN_INSTALL_PATH' in os.environ:
        default_install_path = os.environ['PRNN_INSTALL_PATH']

    vars.Add(PathVariable('install_path', 'The prnn install path',
        default_install_path, PathVariable.PathIsDirCreate))

    vars.Add(BoolVariable('install', 'Include prnn install path in default '
        'targets that will be built and configure to install in the '
        'install_path (defaults to false unless one of the targets is '
        '"install")', 0))

    # add a variable to handle cuda install path
    cuda_path = "/usr/local/cuda"

    if 'CUDA_PATH' in os.environ:
        cuda_path = os.environ['CUDA_PATH']

    vars.Add(PathVariable('cuda_path', 'Cuda toolkit install path', cuda_path,
        PathVariable.PathAccept))

    # add a variable to handle cuda architecture
    default_cuda_arch = 'sm_30'

    if 'CUDA_ARCH' in os.environ:
        default_cuda_arch = os.environ['CUDA_ARCH']

    vars.Add(EnumVariable('cuda_arch', 'Cuda architecture', default_cuda_arch,
        allowed_values = ('sm_30', 'sm_35', 'sm_50', 'sm_52')))

    # create an Environment
    env = OldEnvironment(ENV = importEnvironment(), \
        tools = getTools(), variables = vars)

    updateEnvironment(env)

    # set the version
    env.Replace(VERSION = "0.1")

    # always link with the c++ compiler
    if os.name != 'nt':
        env['LINK'] = env['CXX']

    # get C compiler switches
    env.AppendUnique(CFLAGS = getCFLAGS(env['mode'], env['Wall'], \
        env['Werror'], env.subst('$CC')))

    # get CXX compiler switches
    env.AppendUnique(CXXFLAGS = getCXXFLAGS(env['mode'], env['Wall'], \
        env['Werror'], env.subst('$CXX')))

    # get linker switches
    env.AppendUnique(LINKFLAGS = getLINKFLAGS(env['mode'], env.subst('$LINK')))

    # Install paths
    if env['install']:
        env.Replace(INSTALL_PATH = os.path.abspath(env['install_path']))
    else:
        env.Replace(INSTALL_PATH = os.path.abspath('.'))

    # get libc++
    if env['CXX'] == 'c++':
        env.AppendUnique(CPPPATH = getLibCXXPaths()[0])

    # set extra libs
    env.Replace(EXTRA_LIBS=getExtraLibs(env))

    # set the build path
    env.Replace(BUILD_ROOT = str(env.Dir('.').abspath))
    env.AppendUnique(CPPPATH = os.path.join(env['BUILD_ROOT'], 'include'))

    # set prnn include path
    if env['install']:
        env.AppendUnique(LIBPATH = os.path.abspath(os.path.join(env['install_path'], 'lib')))
    else:
        env.AppendUnique(LIBPATH = os.path.abspath('.'))

    # we need librt on linux
    if sys.platform == 'linux2':
        env.AppendUnique(EXTRA_LIBS = ['-lrt'])

    # we need libdl on max and linux
    if os.name != 'nt':
        env.AppendUnique(EXTRA_LIBS = ['-ldl'])

    # generate help text
    Help(vars.GenerateHelpText(env))

    return env


