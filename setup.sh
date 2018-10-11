sudo pip3 install virtualenv
virtualenv venv
source venv/bin/activate
os=$(uname)
if [ $os = "Linux" ]; then
    sudo apt install poppler-utils
    sudo apt install libpulse-dev
    sudo apt install swig
    pip3 install -r requirements.txt
fi
if [ $os = "Darwin" ]; then
    brew install poppler-utils
    brew install swig
    git clone --recursive https://github.com/bambocher/pocketsphinx-python
    cd pocketsphinx-python
    git checkout b969619
    sed -i -e 's/<al.h>/<OpenAL\/al.h>/g' deps/sphinxbase/src/libsphinxad/ad_openal.c
    sed -i -e 's/<alc.h>/<OpenAL\/alc.h>/g' deps/sphinxbase/src/libsphinxad/ad_openal.c
    python3 setup.py install
    pip3 install -r requirements
fi
