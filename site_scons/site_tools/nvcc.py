
"""SCons.Tool.nvcc

Tool-specific initialization for NVIDIA CUDA Compiler.
There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
from __future__ import print_function
import SCons.Tool
import SCons.Scanner.C
import SCons.Defaults
import os
import platform
import subprocess
import re

def find_file(paths, filename):
    for path in paths:
        if os.path.isfile(os.path.join(path, filename)):
            return path

    return None

def get_library_extension():
  if platform.system() == 'Darwin':
    return '.dylib'
  else:
    return '.so'

def get_cuda_paths(cuda_path = None):
  """Determines CUDA {bin,lib,include} paths

  returns (bin_path,lib_path,inc_path)
  """

  # determine defaults
  if os.name == 'posix':
    lib = 'cudart'
    paths = [cuda_path, '/tools/cuda', '/usr/local/cuda']
    base_path = find_file(paths, ('lib64/lib%s' + get_library_extension()) % lib)

    # OSX still uses lib, not lib64
    if base_path == None:
      base_path = find_file(paths, ('lib/lib%s' + get_library_extension()) % lib)

    if base_path == None:
        base_path = find_file(paths, ('lib/lib%s' + get_library_extension()) % lib)
    if base_path == None:
        base_path = "/usr/local/cuda"
    bin_path = os.path.join(base_path, 'bin')
    lib_path = os.path.join(base_path, 'lib')
    inc_path = os.path.join(base_path, 'include')
  else:
    raise ValueError, 'Error: unknown OS.  Where is nvcc installed?'

  if platform.platform()[:6] != 'Darwin' and \
      platform.machine()[-2:] == '64':
    lib_path += '64'

  # override with environement variables
  if 'CUDA_BIN_PATH' in os.environ:
    bin_path = os.path.abspath(os.environ['CUDA_BIN_PATH'])
  if 'CUDA_LIB_PATH' in os.environ:
    lib_path = os.path.abspath(os.environ['CUDA_LIB_PATH'])
  if 'CUDA_INC_PATH' in os.environ:
    inc_path = os.path.abspath(os.environ['CUDA_INC_PATH'])

  return (bin_path,lib_path,inc_path)

CUDASuffixes = ['.cu', '.cpp']

# make a CUDAScanner for finding #includes
# cuda uses the c preprocessor, so we can use the CScanner
CUDAScanner = SCons.Scanner.C.CScanner()

def get_architecture_major(arch):
    return arch[3]

def add_common_nvcc_variables(env):
  """
  Add underlying common "NVIDIA CUDA compiler" variables that
  are used by multiple builders.
  """

  # "NVCC common command line"
  if not env.has_key('_NVCCCOMCOM'):
    # nvcc needs '-I' prepended before each include path, regardless of platform
    env['_NVCCWRAPCPPPATH'] = '${_concat("-I ", CPPPATH, "", __env__, RDirs)}'
    # prepend -Xcompiler before each flag
    env['_NVCCWRAPCFLAGS'] =     '${_concat("-Xcompiler ", CFLAGS,     "", __env__)}'
    env['_NVCCWRAPSHCFLAGS'] =   '${_concat("-Xcompiler ", SHCFLAGS,   "", __env__)}'
    env['_NVCCWRAPCCFLAGS'] =   '${_concat("-Xcompiler ", CCFLAGS,   "", __env__)}'
    env['_NVCCWRAPSHCCFLAGS'] = '${_concat("-Xcompiler ", SHCCFLAGS, "", __env__)}'
    # assemble the common command line
    env['_NVCCCOMCOM'] = '${_concat("-Xcompiler ", CPPFLAGS, "", __env__)} $_CPPDEFFLAGS $_NVCCWRAPCPPPATH'

def add_nvcc_flags(env):
  env['NVCCFLAGS'] = SCons.Util.CLVar('')
  env['SHNVCCFLAGS'] = SCons.Util.CLVar('') + ' -shared'

  if 'CUDA_NVCC_FLAGS' in os.environ:
    for flag in os.environ['CUDA_NVCC_FLAGS'].split():
      env.AppendUnique(NVCCFLAGS = flag)

  arch = env['cuda_arch']
  if not arch:
    arch = 'sm_30'

  gencode = '-gencode=arch='+arch.replace("sm","compute")+',code='+arch
  env.AppendUnique(NVCCFLAGS = gencode)

  if platform.platform()[:6] == 'Darwin':
    if platform.machine()[-2:] == '64':
      env.AppendUnique(NVCCFLAGS = '-m64')
    else:
      env.AppendUnique(NVCCFLAGS = '-m32')

  if env['mode'] == 'debug':
    env.AppendUnique(NVCCFLAGS = '-G')
    pass

  env.AppendUnique(NVCCFLAGS = '-std=c++11')
  env.AppendUnique(NVCCFLAGS = '-D CUDA_ARCH_MAJOR=' + get_architecture_major(arch))
  env.AppendUnique(NVCCFLAGS = '-Xcompiler=-Wno-unused-function')
  env.AppendUnique(NVCCFLAGS = '-Xcompiler=-Wno-unused-private-field')
  #env.AppendUnique(NVCCFLAGS = '-Xcompiler=-Wno-unused-local-typedef')

def cuda_exists(env):
    if not env['with_cuda']:
        return False

    return os.path.exists(env['cuda_path'])

def generate_dummy(env):
  bld = SCons.Builder.Builder(action = SCons.Defaults.Copy('$TARGET', '$SOURCE'),
                  suffix = '.cpp',
                  src_suffix = '.cu')
  env['BUILDERS']['CUDASharedObject'] = bld

def generate(env):
  """
  Add Builders and construction variables for CUDA compilers to an Environment.
  """

  if not cuda_exists(env):
    print('Failed to build NVCC tool')
    generate_dummy(env)
    return

  # create a builder that makes PTX files from .cu files
  ptx_builder = SCons.Builder.Builder(action = '$NVCC -ptx $NVCCFLAGS $_NVCCWRAPCFLAGS $NVCCWRAPCCFLAGS $_NVCCCOMCOM $SOURCES -o $TARGET',
                                      emitter = {},
                                      suffix = '.ptx',
                                      src_suffix = CUDASuffixes)
  env['BUILDERS']['PTXFile'] = ptx_builder

  print('Building NVCC tool')

  # create builders that make static & shared objects from .cu files
  static_obj, shared_obj = SCons.Tool.createObjBuilders(env)

  for suffix in CUDASuffixes:
    # Add this suffix to the list of things buildable by Object
    static_obj.add_action(suffix, '$NVCCCOM')
    shared_obj.add_action(suffix, '$SHNVCCCOM')
    static_obj.add_emitter(suffix, SCons.Defaults.StaticObjectEmitter)
    shared_obj.add_emitter(suffix, SCons.Defaults.SharedObjectEmitter)
    env['BUILDERS']['CUDASharedObject'] = shared_obj

    # Add this suffix to the list of things scannable
    SCons.Tool.SourceFileScanner.add_scanner(suffix, CUDAScanner)

  add_common_nvcc_variables(env)

  # set the "CUDA Compiler Command" environment variable
  env['NVCC'] = 'nvcc'
  env['SHNVCC'] = 'nvcc'

  # set the include path, and pass both c compiler flags and c++ compiler flags
  add_nvcc_flags(env)

  # 'NVCC Command'
  env['NVCCCOM']   = '$NVCC -o $TARGET -c $_NVCCWRAPCFLAGS $NVCCWRAPCCFLAGS $_NVCCCOMCOM $NVCCFLAGS $SOURCES'
  env['SHNVCCCOM'] = '$SHNVCC -o $TARGET -c $SHNVCCFLAGS $_NVCCWRAPSHCFLAGS $_NVCCWRAPSHCCFLAGS $_NVCCCOMCOM $NVCCFLAGS $SOURCES'

  # XXX add code to generate builders for other miscellaneous
  # CUDA files here, such as .gpu, etc.

  (bin_path,lib_path,inc_path) = get_cuda_paths(env['cuda_path'])
  env.Append(LIBPATH = [lib_path])
  env.Append(RPATH = [lib_path])
  env.Append(CPPPATH = [inc_path])
  env.PrependENVPath('PATH', bin_path)

def exists(env):
  return env.Detect('nvcc')


