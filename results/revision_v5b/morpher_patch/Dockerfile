FROM quotientflow-morpher-v5-original-patched:latest
COPY Morpher_CGRA_Mapper /home/user/morpher/Morpher_CGRA_Mapper
COPY Morpher_DFG_Generator /home/user/morpher/Morpher_DFG_Generator
RUN cd /home/user/morpher/Morpher_CGRA_Mapper && rm -rf build && mkdir build && cd build && cmake .. && make all -j2
