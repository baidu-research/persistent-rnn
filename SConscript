################################################################################
#    \file   SConscript
#    \brief  The SCons master build script for the persistent RNN kernels.
################################################################################

import os

def toSharedObject(filename):
    (root, ext) = os.path.splitext(str(filename))
    return root + '.os'

# set environment variables
prnn_args = ARGUMENTS
prnn_cmd_line_targets = COMMAND_LINE_TARGETS
prnn_path_to_root_dir = ".."

# try to import an environment
env = BuildEnvironment()
env['PRNN_PATH'] = Dir('.').abspath

env['path_to_root_dir'] = os.path.abspath(prnn_path_to_root_dir)

# find all source files in the source tree
directories = ['src', 'src/detail/matrix', 'src/detail/parallel', 'src/detail/util',
    'src/detail/rnn']

source_directories = directories
cuda_source_directories = directories

extensions = ['*.cpp']

sources = []
for dir in source_directories:
    for ext in extensions:
        regexp = os.path.join(dir, ext)
        sources.extend(env.Glob(regexp))

for dir in cuda_source_directories:
    regexp = os.path.join(dir, '*.cu')
    cuda_sources = env.Glob(regexp)
    for cuda_source in cuda_sources:
        sources.append(env.CUDASharedObject(cuda_source))

# create the library
libprnn = env.SharedLibrary('prnn', sources, LIBS=env['EXTRA_LIBS'])
prnn_libs = ['prnn'] + env['EXTRA_LIBS']

if env['install']:
    libprnn = env.Install(os.path.join(env['install_path'], "lib"),
        libprnn)

env['PRNN_LIBS'] = prnn_libs

# create the programs
programs = []

programs.extend(SConscript('test/SConscript', exports='env'))
programs.extend(SConscript('benchmark/SConscript', exports='env'))

for program in programs:
    env.Depends(program, libprnn)

# install it all
if env['install']:
    installed   = []
    executables = []

    print 'Installing Persistent RNN Kernels'

    for program in programs:
        executables.append(env.Install(
            os.path.join(env['install_path'], "bin"), program))

    # install headers
    header_directories = ['include/prnn/detail/matrix', 'include/prnn/detail/util',
        'include/prnn/detail/parallel', 'include/prnn/detail/rnn', 'include/prnn/detail/util']
    header_extensions = ['*.h']

    headers = []
    for dir in header_directories:
        for ext in header_extensions:
            regexp = os.path.join(dir, ext)
            headers.extend(env.Glob(regexp))

    for header in headers:
        (directoryPath, headerName) = os.path.split( \
            os.path.relpath(str(header), env['path_to_root_dir']))
        installed.append(env.Install(os.path.join( \
            env['install_path'], "include", directoryPath), header))

    # set permissions
    for i in executables:
        env.AddPostAction(i, Chmod(i, 0755))
    for i in installed:
        env.AddPostAction(i, Chmod(i, 0644))

    # Run the install rules by default
    install_alias = env.Alias('install', env['install_path'])
    Default(install_alias)





