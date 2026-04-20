pip uninstall -y transformer_engine
rm -rf build transformer_engine.egg-info/
export TE_FL_SKIP_CUDA=1
export NVTE_FRAMEWORK=pytorch
pip install --no-build-isolation . -vvv
