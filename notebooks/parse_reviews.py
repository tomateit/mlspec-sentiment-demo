import requests
import json
import sqlite3
import time
import random
import re
from tqdm import tqdm 

regexp_catalog_id = re.compile(r"(?:_)\d{5,20}(?:-catalog)")
regexp_digits = re.compile(r"(?:\D*)(\d{5,20})(?:\D*)")

class Parser():
    def __init__(self, database_name=None):
        if not database_name:
            print("No database name provided. Creating...")
            database_name = f"reviews_{random.randint(100,999)}.db"
        self.conn = sqlite3.connect(database_name)
        print(f"Connected to database {database_name}")

    def _create_tables(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE catalog_ids (catalog_id text)''')
        c.execute('''CREATE TABLE reviews (content text, label text)''')
        self.conn.commit()
        print("Tables created")

    def _push_id_to_db(self, cid):
        c = self.conn.cursor()
        c.execute(f"INSERT INTO catalog_ids VALUES ('{str(cid)}')")
        self.conn.commit()

    def _push_review_to_db(self, text, label):
        c = self.conn.cursor()
        c.execute(f"INSERT INTO reviews VALUES ('{text}','{label}')")
        self.conn.commit()

    def _get_all_catalog_ids(self):
        c = self.conn.cursor()
        catalog_ids = list(c.execute('''SELECT DISTINCT catalog_id FROM catalog_ids'''))
        catalog_ids = list(map(lambda x: x[0], catalog_ids))
        return catalog_ids

    def _parse_catalog_ids(self):
        for n_page in range(1,100):
            print(f"Doing {n_page} catalog page")
            response_page = requests.get(f"https://hi-tech.mail.ru/ajax/mobile-catalog/?sort=popular&sort_order=desc&page={n_page}")
            if (response_page.status_code != 200):
                print(f"status is not 200 for {response_page.url}: {response_page.status_code}")
                break
            cids = regexp_catalog_id.findall(response_page.text)
            for cid in cids:
                cid_ = regexp_digits.match(cid).group(1)
                self._push_id_to_db(cid_)
            time.sleep(random.randint(1,5))


    def _parse_reviews(self, cids=None):
        if not cids:
            cids = self._get_all_catalog_ids()

        for cid in tqdm(cids):
            try_pages = True
            page_increment = 1
            while (try_pages):
                page = requests.get(f"https://hi-tech.mail.ru/ajax/{cid}-catalog/feedback/list/?page={page_increment}")
                page_increment += 1
                try_pages = self._process_page(page)
                time.sleep(random.randint(1,5))
    
    def _process_page(self, response_page):
        try:
            # check status
            if (response_page.status_code != 200):
                print(f"status is not 200 for {response_page.url}: {response_page.status_code}")
                return False
            # check content
            page_data = json.loads(response_page.text).get("data", [])
            if (not page_data):
                print("page does not contain data")
                return False
            # try extraction
            self._extract_reviews_from_data(page_data)
            print("extracted from ", response_page.url)
            # extraction was successfull, maybe try next page
            return True
        except:
            print("exception occured")
            return False
    
    def _extract_reviews_from_data(self, data):
        for rev in data:
            text_ = {}
            for t in rev['text']:
                text_[t["title"]] = t["text"]
            
            label = "pos" if rev["vote_type"] == "positive" else "neg"
            # я буду отдавать позитивной части отзыва если он положительный и негативной - если отрицательный
            # не буду - базовые метки очень плохие
            # if (label=="pos"):
            #     text = f"{text_['Плюсы']} {text_['Впечатления']}"
            # else:
            #     text = f"{text_['Минусы']} {text_['Впечатления']}"
            # text = f"{text_['Плюсы']}. {text_['Минусы']}. {text_['Впечатления']}."
            text = f"Плюсы: {text_['Плюсы']}. Минусы: {text_['Минусы']}. Впечатления: {text_['Впечатления']}."
    #         print(label, text)
            self._push_review_to_db(text, label)

    def main(self):
        self._create_tables()
        self._parse_catalog_ids()
        self._parse_reviews()
        print("Parsing finished.")



if __name__ == "__main__":
    try:
        _p = Parser()
        _p.main()
    except Exception e:
        print(e)
    finally:
        _p.conn.close()