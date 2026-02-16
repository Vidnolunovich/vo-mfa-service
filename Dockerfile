# MFA Service Dockerfile
# Builds a container with Montreal Forced Aligner and pretrained models

FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    wget \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:$PATH"

# Install MFA (latest compatible version, let conda resolve dependencies)
RUN conda create -n mfa python=3.10 -y \
    && conda run -n mfa conda install -c conda-forge montreal-forced-aligner -y \
    && conda clean -afy

# Make mfa environment default
ENV PATH="/opt/conda/envs/mfa/bin:$PATH"
ENV CONDA_DEFAULT_ENV=mfa

# Verify MFA works
RUN mfa version

# Download pretrained models (при билде, не при старте!)
# English
RUN mfa model download acoustic english_us_arpa \
    && mfa model download dictionary english_us_arpa

# Russian
RUN mfa model download acoustic russian_mfa \
    && mfa model download dictionary russian_mfa

# Spanish
RUN mfa model download acoustic spanish_mfa \
    && mfa model download dictionary spanish_mfa

# German
RUN mfa model download acoustic german_mfa \
    && mfa model download dictionary german_mfa

# Portuguese
RUN mfa model download acoustic portuguese_mfa \
    && mfa model download dictionary portuguese_mfa

# Set MFA environment
ENV MFA_ROOT_DIR=/root/Documents/MFA

# Install Python dependencies for FastAPI
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY aligner.py .
COPY rms_refiner.py .

# Expose port
EXPOSE 8080

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
