## Path
export LOOPY_PATH="$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)"
export PATH="$LOOPY_PATH/bin:$PATH"
export PATH="$LOOPY_PATH/LOOPY_bin:$PATH"
export PYTHONPATH="$LOOPY_PATH/LiCSBAS_lib:$PYTHONPATH"
export PYTHONPATH="$LOOPY_PATH/LOOPY_lib:$PYTHONPATH"

