# MFA Service Dockerfile
# Builds a container with Montreal Forced Aligner and pretrained models

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install conda (required for MFA)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Install MFA via conda
RUN conda install -c conda-forge montreal-forced-aligner=3.1.0 -y \
    && conda clean -afy

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

# Install Python dependencies
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
