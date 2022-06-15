#!/bin/bash

pip3 install gdown

mkdir -p data/anime
mkdir data/lastfm

# book crossing
gdown 154kLiPBBuu1d5VIxxmHUvAceOIlPsib9
unzip bookcrossing.zip
mv bookcrossing data/
mv bookcrossing.zip data/bookcrossing/

# Anime
gdown 1yrqEp1L2y1cppLYsVDMuE146KgI2WlMk
mv archive.zip data/anime/
unzip data/anime/archive.zip
mv anime.csv items.csv
mv *.csv data/anime/


# last.fm
wget http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz
tar -xvzf lastfm-dataset-360K.tar.gz
mv *.gz lastfm-dataset-360K/* data/lastfm
rm -r lastfm-dataset-360K