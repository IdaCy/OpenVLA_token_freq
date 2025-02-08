
## create overlay:
singularity overlay create --size 4096 openvla_overlay.ext3


## start a singularity shell with overlay containers/overlays/openvla_overlay.ext3:

singularity shell --nv --overlay containers/overlays/openvla_overlay.ext3 containers/ctopenvla.sif


## install missing things:

pip install rlds


## check success:

python -c "import rlds; print('rlds is installed')"

exit

