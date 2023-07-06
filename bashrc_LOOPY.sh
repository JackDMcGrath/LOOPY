## Path
# Deactivated LiCSBAS numbered commands - now sourcing from QiCSBAS
export LOOPY_PATH="$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)"
export PATH="$LOOPY_PATH/LOOPY_bin:$PATH"
export PYTHONPATH="$LOOPY_PATH/LOOPY_lib:$PYTHONPATH"

#export PATH="$LOOPY_PATH/bin:$PATH"
#export PYTHONPATH="$LOOPY_PATH/LiCSBAS_lib:$PYTHONPATH"

# Deactivated for sourcing independent LiCSBAS_qi
#export LICSBAS_PATH="/nfs/a285/homes/eejdm/software/LiCSBAS_qi"
#export PATH="$LICSBAS_PATH/bin:$PATH"
#export PYTHONPATH="$LICSBAS_PATH/LiCSBAS_lib:$PYTHONPATH"

#export LICSBAS_PATH="/nfs/a285/homes/eejdm/software/LiCSBAS_qi"
export PATH="$LOOPY_PATH/bin:$PATH"
export PYTHONPATH="$LOOPY_PATH/LiCSBAS_lib:$PYTHONPATH"
