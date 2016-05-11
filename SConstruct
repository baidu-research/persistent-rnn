################################################################################
#	\file   SConstruct
#	\brief  The SCons master build script for the persistent rnn kernels.
################################################################################

import os

def mkdir(name):
	if not os.path.isdir(name):
		os.mkdir(name)

if ARGUMENTS.get('mode', 'debug') == 'release':
	mkdir('.release_build')
	SConscript('SConscript', variant_dir='.release_build', duplicate=0,
		exports={'mode':'release'})
else:
	mkdir('.debug_build')
	SConscript('SConscript', variant_dir='.debug_build',   duplicate=0,
		exports={'mode':'debug'})



