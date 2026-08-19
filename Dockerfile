FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS xkv-builder

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch==2.10.0
RUN git clone https://github.com/abdelfattah-lab/xKV.git /tmp/xkv \
    && git -C /tmp/xkv checkout 05d91ecb0d698279aa4220fda5d7a5108036d692 \
    && git -C /tmp/xkv submodule update --init --depth 1 3rdparty/cutlass
RUN cd /tmp/xkv/efficiency \
    && TORCH_CUDA_ARCH_LIST=9.0 python3 setup.py build_ext --inplace \
    && mkdir -p /opt/xkv_kernels \
    && cp ops/_shadowkv*.so /tmp/xkv/LICENSE /opt/xkv_kernels/

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Install Python and pip inside the container base
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /workspace

# Copy your requirements file and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY --from=xkv-builder /opt/xkv_kernels /workspace/efficiency/ops

# Copy the rest of your repository code into the container
COPY . .