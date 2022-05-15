from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

# APIの情報

key = "7bf1337b1b8d39a3e4a81895de15c366"
secret = "acf7156b925cd48d"
wait_time = 1

#保存フォルダの指定
animalname = sys.argv[1]
savedir = "./" + animalname # 現在のディレクトリにあるanimalname

flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    text = animalname,
    per_page = 400,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, licence'
)

photos = result['photos']
#返り値を表示する
#pprint(photos)

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'
    if os.path.exists(filepath): continue
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)
