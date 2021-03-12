# Собирает документацию и пакет и кладет их в указанную папку

if [ -z "$1" ]
  then
    echo "Provide path to destination folder as an argument"
    exit 1
fi

mkdir -p $1
make clean html --directory=docs
cp -R docs/_build/html/. $1/docs
poetry build
cp -R dist/. $1/dist
rsync -a --exclude=".*" experiments/ $1/examples