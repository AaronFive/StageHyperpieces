import os

import pandas as pd
import requests

# URL of the Dracor API endpoint for the French corpus
language_string = 'fre'
base_url = f'https://dracor.org/api/v1/corpora/{language_string}'

# Directory to save the downloaded CSV files
save_dir = 'dracor_data'
os.makedirs(save_dir, exist_ok=True)

# Fetching data from Dracor API
resp = requests.get(base_url)

# The pre-computed character csvs from Dracor do not align with the actual speakers when parsing the texts manually.
# Below are two functions, one that get the character csvs from te Dracor API, and one that cmputez them by hand
def get_all_characters_csv(response):
    """Downloads all the characters csv from Dracor"""
    failed = None
    if response.status_code == 200:
        failed = []
        data = response.json()
        for p in data['plays']:
            name = p['name']
            print(name)
            dracor_id = p['id']
            response_play = requests.get(f'{base_url}/plays/{name}/characters')
            if response_play.status_code == 200:
                response_play = response_play.json()
                play_df = pd.DataFrame(response_play)
                play_df.to_csv(f'{save_dir}\\{dracor_id}_{name}.csv', index=False)
            else:
                print(f"Warning, request error for {name}")
                failed.append(dracor_id)
    print(f"Failed {failed}")


if __name__ == "__main__":
    get_all_characters_csv(resp)
