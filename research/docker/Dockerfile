######
# Combination of Jupyter / Datascience 
# and k8s container from OU
# https://gitlab.nrp-nautilus.io/oulib/kubernetes-tools-in-jupyter/-/blob/main/kubernetes-notebook/Dockerfile
########
FROM jupyter/datascience-notebook:python-3.8

USER root


# Install all OS dependencies for fully functional notebook server
RUN apt update -y && \
    apt install -y --no-install-recommends \
    # Common useful utilities
    git \
    nano-tiny \
    tzdata \
    unzip \
    vim-tiny \
    # git-over-ssh
    openssh-client \
    # TODO: check if these are needed and describe
    libsm6 \
    libxext-dev \
    libxrender1 \
    lmodern \
    netcat \
    tmux \
    curl \
    jq 
# && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /k8s

# Install Kubernets kubectl tool
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    curl -LO "https://dl.k8s.io/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256" && \
    bash -c 'echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check' && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Insall k9s tool
RUN curl -LO "https://github.com/derailed/k9s/releases/download/v0.27.4/k9s_Linux_amd64.tar.gz" && \
    bash -c 'echo "e507831ebd5f9b8c0380f212669f352c6e34cb760c916b498babae8be83c4392  k9s_Linux_amd64.tar.gz" | sha256sum --check' && \
    tar xzf k9s_Linux_amd64.tar.gz && \
    install -o root -g root -m 0755 k9s /usr/local/bin/k9s

# Install Helm tool
RUN curl -LO "https://get.helm.sh/helm-v3.12.0-linux-amd64.tar.gz" && \
    bash -c 'echo "da36e117d6dbc57c8ec5bab2283222fbd108db86c83389eebe045ad1ef3e2c3b  helm-v3.12.0-linux-amd64.tar.gz" | sha256sum --check' && \
    tar xzf helm-v3.12.0-linux-amd64.tar.gz && \
    install -o root -g root -m 0755 linux-amd64/helm /usr/local/bin/helm

WORKDIR "${HOME}"

# Create alternative for nano -> nano-tiny
RUN update-alternatives --install /usr/bin/nano nano /bin/nano-tiny 10

# Install our deps
COPY /requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt && rm /tmp/requirements.txt
RUN pip3 install --extra-index-url https://gitlab.nrp-nautilus.io/api/v4/projects/2953/packages/pypi/simple nautiluslauncher


# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}
