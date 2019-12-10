import sys
import requests

from sponge_bob_magic.data_loader.archives import extract, delete


def download_dataset(link, archive_name):

    try:
        download_url(link, archive_name)
        extract(archive_name)
        delete(archive_name)
        print('Done\n')

    except Exception as e:
        print(e)


def download_url(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                sys.stdout.flush()

    sys.stdout.write('\n')
