import pandas as pd

import json

def get_author(row):
    author = row['author']
    # Si c'est une string, on tente de la parser en dict
    if isinstance(author, str):
        try:
            author = json.loads(author)
        except Exception:
            author = {}

    if 'name' in author and author['name']:
        return author['name']
    elif 'twitter' in author and author['twitter']:
        return author['twitter'].replace('@', '')
    else:
        url = row['url']
        if 'medium.com/@' in url:
            return url.split('medium.com/@')[1].split('/')[0]
        elif 'medium.com/' in url:
            return url.split('medium.com/')[1].split('/')[0]
        else:
            return 'unknown'
