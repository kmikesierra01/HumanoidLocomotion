@echo off
rem One-time setup: new venv + Isaac Sim 4.5.0 + Isaac Lab v2.1.1 (rsl_rl).
rem NVIDIA driver must stay at 591.86 (595+ crashes Isaac Sim 4.5 with rtx.scenedb access violation).
rem Install locations: set VENV / LAB before running to change them (e.g. set VENV=D:\venv-lab).
if not defined VENV set VENV=C:\Storage\IsaacSim\venv-lab
if not defined LAB set LAB=C:\Storage\IsaacSim\IsaacLab

if not exist %LAB% git clone --depth 1 --branch v2.1.1 https://github.com/isaac-sim/IsaacLab.git %LAB% || exit /b 1
if not exist %VENV% py -3.10 -m venv %VENV% || exit /b 1
call %VENV%\Scripts\activate.bat
rem isaaclab.bat prefers conda python if CONDA_PREFIX is set
set CONDA_PREFIX=

python -m pip install --upgrade pip || exit /b 1
rem Isaac Lab 2.1.1 requires torch 2.7.0+cu128 (isaaclab.bat reinstalls it otherwise); isaacsim-core only needs >=2.5.1
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 || exit /b 1
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com || exit /b 1
rem flatdict 4.0.1 (isaaclab dep) imports pkg_resources, gone from the latest setuptools pip uses for isolated builds
rem -> build it with the venv's setuptools 65. Must succeed first: isaaclab.bat misdetects python after any failure.
pip install wheel || exit /b 1
pip install flatdict==4.0.1 --no-build-isolation || exit /b 1
pushd %LAB%
rem full path: cmd may not search the current dir (NoDefaultCurrentDirectoryInExePath)
call %LAB%\isaaclab.bat --install rsl_rl || exit /b 1
popd
rem sim2sim deps: dry run showed only new packages (no version changes), so same venv
pip install mujoco==3.17.0 onnxruntime==1.23.2 || exit /b 1
python -c "import torch, isaaclab, rsl_rl; print('SETUP OK', torch.__version__, torch.cuda.is_available())"
