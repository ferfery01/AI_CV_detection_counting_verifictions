echo "******** Installing required dependencies ********"
pip install -r requirements-training.txt

echo "**** Installing project and dev dependencies *****"
pip install -e .
