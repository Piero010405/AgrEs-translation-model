# Usa una imagen base de PyTorch con soporte CUDA
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04
# Evita preguntas durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Actualiza y instala dependencias básicas
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Establece python3 como predeterminado
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copia requirements y los instala
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# Crea directorios de trabajo
WORKDIR /workspace

# Copia scripts y dataset al contenedor
COPY scripts/ ./scripts/
COPY data/ ./data/

# Comando por defecto
CMD ["bash"]
