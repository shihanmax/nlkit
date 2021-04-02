# !/bin/bash
bash build.sh
pip uninstall nlkit
pip install dist/nlkit*.whl