FROM ubuntu:18.04

# Install prerequisite packages
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
      jupyter-notebook

# User configuration
ARG USERNAME=jupyter
RUN useradd -m -s /bin/bash ${USERNAME}
USER ${USERNAME}

# Jupyter configuration
RUN jupyter notebook --generate-config \
 && mkdir -p /home/${USERNAME}/jupyter-working \
 && sed -i.back \
    -e "s:^#c.NotebookApp.token = .*$:c.NotebookApp.token = u'':" \
    -e "s:^#c.NotebookApp.ip = .*$:c.NotebookApp.ip = '*':" \
    -e "s:^#c.NotebookApp.open_browser = .*$:c.NotebookApp.open_browser = False:" \
    -e "s:^#c.NotebookApp.notebook_dir = .*$:c.NotebookApp.notebook_dir = '/home/${USERNAME}/jupyter-working':" \
    /home/${USERNAME}/.jupyter/jupyter_notebook_config.py

# Expose container ports
EXPOSE 8888

# Boot process
CMD jupyter notebook
