python3 model/setup-build.py build_ext --inplace
docker build -t openagua/waterlp-mcma .
docker push openagua/waterlp-mcma
