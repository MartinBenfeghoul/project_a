#!/usr/bin/bash
set -euo pipefail

user=$1

# create authorized_keys dir with correct permissions
sudo -u $user bash -lc 'umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys'

echo "Paste the public key on the following lines and press Enter then Ctrl+D"
sudo -u $user bash -lc 'cat >> ~/.ssh/authorized_keys'

echo "Make sure this is the correct public key:"
sudo -u $user bash -lc 'wc -l ~/.ssh/authorized_keys; tail -n 2 ~/.ssh/authorized_keys'

# set .ssh dir permissions
sudo chmod 700 /home/$user/.ssh
sudo chmod 600 /home/$user/.ssh/authorized_keys
sudo chown -R $user:ma-group /home/$user/.ssh

