# !bin/bash
echo "cleaning..."
rm -rf build
rm -rf dist
rm -rf *.egg-info

echo "building..."
python setup.py sdist bdist_wheel