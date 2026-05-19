import logging
from xml.etree import ElementTree

import requests

from utils.mining import logger, get_module_data, save_all_data_item_files

GESETZE_TOC_URL = "https://www.gesetze-im-internet.de/gii-toc.xml"


class Miner:
    @staticmethod
    def mine_gesetze_im_internet(count: int | None = None):
        module_name = "gesetze_im_internet"
        existing = get_module_data(module_name)

        if len(existing) > 0:
            logger.info(f"Found {len(existing)} existing laws, skipping download")
            return existing

        logger.info("Downloading Gesetze-im-Internet TOC ...")
        resp = requests.get(GESETZE_TOC_URL, timeout=60)
        tree = ElementTree.fromstring(resp.content)
        items = tree.findall("item")

        data_items = []
        total = count or len(items)

        for i in range(total):
            title_elem = items[i][0]
            link_elem = items[i][1]
            title = title_elem.text if title_elem is not None else None
            link = link_elem.text if link_elem is not None else None

            if not title or not link:
                continue

            try:
                logger.info(f"Downloading [{i}/{total}] {title}")
                zip_resp = requests.get(link, timeout=120)
                data_items.append({
                    "id": i,
                    "title": title.strip(),
                    "file": zip_resp.content,
                })
            except Exception as e:
                logger.error(f"Failed to download {title}: {e}")

        save_all_data_item_files(module_name, data_items)
        logger.info(f"Downloaded {len(data_items)} laws from Gesetze-im-Internet")
        return data_items
