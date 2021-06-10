FROM continuumio/anaconda3:5.0.1

# install sys requirements
RUN apt-get update
RUN apt-get install -y build-essential git

# clone in repo
RUN git clone https://github.com/DerThorsten/opengm_python

# update conda
RUN conda update --yes conda

# go into repo
RUN cd opengm_python && conda env create -f opengm_python-dev-requirements.yml

# activate
RUN echo "source activate opengm_python-dev-requirements" > ~/.bash
ENV PATH /opt/conda/envs/opengm_python-dev-requirements/bin:$PATH

# activate
#RUN conda activate opengm_python-dev-requirements

# make build dir
RUN cd opengm_python && mkdir build

# run cmake
RUN  cd opengm_python/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$(conda info --base)/envs/opengm_python-dev-requirements

# build and install project
RUN cd opengm_python/build && \
    make -j2 install