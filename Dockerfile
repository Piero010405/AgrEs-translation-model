# Imagen base de PyTorch estable con CUDA 12.1 y cuDNN 8
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Dependencias básicas
RUN apt-get update && apt-get install -y \
    git wget python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Requisitos
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia scripts y datos
COPY scripts/ ./scripts/
COPY data/ ./data/

# Comando por defecto
CMD ["bash"]
