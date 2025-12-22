pip install torch torchvision
pip install pillow
pip install tensorflow
pip install matplotlib
pip install prefetch_generator
pip install gensim

cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

cd ../nlg-eval
python setup.py build; python setup.py install;
nlg-eval --help
cd ..