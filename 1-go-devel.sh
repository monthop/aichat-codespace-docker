sudo mkdir /tmp/pn
sudo chown codespace:codespace /tmp/pn
sudo mkdir /tmp/transformers
sudo chown codespace:codespace /tmp/transformers
sudo mkdir /tmp/datasets
sudo chown codespace:codespace /tmp/datasets
docker run --rm -it --user 1000 -v /tmp/pn:/home/pn -v /tmp/datasets:/opt/datasets -v /tmp/transformers:/opt/transformers -v /workspaces/aichat-codespace-docker/devel:/opt/devel nikolaik/python-nodejs:python3.10-nodejs18-bullseye bash
