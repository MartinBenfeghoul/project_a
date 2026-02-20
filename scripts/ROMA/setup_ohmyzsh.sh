#!/usr/bin/bash
set -euo pipefail

# install zsh
sudo apt update
sudo apt install -y zsh

# clone into the right path
git clone https://github.com/ohmyzsh/ohmyzsh.git ~/.oh-my-zsh
cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc
# zsh ~/.oh-my-zsh/oh-my-zsh.sh  # apparently this is unnecessary if .zshrc is in the right place

# download plugins
git clone https://github.com/zsh-users/zsh-autosuggestions \
  ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting \
  ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting
echo "Plugins downloaded."

echo "plugins=(git zsh-autosuggestions zsh-syntax-highlighting)" >> ~/.zshrc

# Make zsh the default shell
ZSH_PATH="$(command -v zsh)" && chsh -s "$ZSH_PATH"
echo "Made zsh the default shell"

# Initialise conda in zsh
bash -lc 'source ~/.bashrc; conda init zsh'
echo "Initialised conda with zsh"

# Customise the zsh configs
echo "Now open ~/.zshrc to further customise the shell."
echo "See comments at the bottom of this script for options I like to add"
echo "..or optionally just copy the .zshrc file in this directory to your user dir"

# # Robbyrussell: replace ✗ with a coloured $ while keeping the theme’s wrapping structure
# ZSH_THEME_GIT_PROMPT_DIRTY="%{$fg[blue]%}) %{$fg[red]%}%1{\$%}%{$reset_color%}"
# ZSH_THEME_GIT_PROMPT_CLEAN="%{$fg[blue]%}) %{$fg[green]%}%1{\$%}%{$reset_color%}"

# # Enable ctrl+backspace
# bindkey '^H' backward-kill-word
# bindkey '^[^H' backward-kill-word

# # Path highlighting styles (tweak to taste)
# ZSH_HIGHLIGHT_STYLES[path]='fg=cyan'
# ZSH_HIGHLIGHT_STYLES[path_prefix]='fg=red'
# ZSH_HIGHLIGHT_STYLES[path_approx]='fg=yellow'
