FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities.
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory.
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

# Download and install Micromamba.
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
  && bash Miniforge3-$(uname)-$(uname -m).sh -b -p "${HOME}/conda"
RUN bash "${HOME}/conda/etc/profile.d/conda.sh" && bash "${HOME}/conda/etc/profile.d/mamba.sh"

#RUN curl -sL https://micro.mamba.pm/api/micromamba/linux-64/1.1.0 \
#  | sudo tar -xvj -C /usr/local bin/micromamba
ENV MAMBA_ROOT_PREFIX=${HOME}/conda \
    CONDA_PREFIX=${HOME}/conda \
    PATH=${HOME}/.local/bin:${HOME}/conda/bin:$PATH

# Set up the base Conda environment by installing PyTorch and friends.
RUN python3 -m pip install --user pipx && python3 -m pipx ensurepath && pipx install poetry

COPY pyproject.toml /app/
RUN poetry install && pip install -U kaleido

# Set the default command to python3.
CMD ["bash"]
