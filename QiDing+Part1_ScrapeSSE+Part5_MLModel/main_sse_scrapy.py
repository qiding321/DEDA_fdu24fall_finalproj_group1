
import os
import requests
from bs4 import BeautifulSoup
import datetime
import my_logger

this_log = my_logger.this_log


def main():
    # p_out_root = r'G:\data\sse_scrapy' + '\\'
    p_out_root = r'/data/sse_scrapy/'
    p_update_flag = p_out_root+'update_flag.txt'
    this_log.add_path(p_out_root+'log.txt')
    time_now = datetime.datetime.now().strftime('%Y%m%d%H%M')
    date = time_now[:8]
    p_out = p_out_root+date+'/'+time_now+'/'
    if not os.path.exists(p_out):
        os.makedirs(p_out)
        this_log.info(f'mkdir {p_out}')

    main_url = 'https://www.sse.net.cn/indexIntro?indexName=intro'
    root_url = 'https://www.sse.net.cn'
    this_log.info(f'connect to {main_url}')
    response = requests.get(main_url)
    this_log.info(f'connection response {response}')
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all the links to other websites
    links = soup.find_all('a', href=True)
    target_links = [link for link in links if "index" in link['href'].lower()]
    this_log.info(f'find links {len(links)}, target links  {len(target_links)}')

    for link in target_links:
        head = link.get_text().strip()
        # Build the full URL of each linked website
        linked_url = link['href']
        if not linked_url.startswith("http"):
            linked_url = root_url + linked_url
        this_log.info(f'access link {linked_url}')

        # Fetch the content of the linked website
        linked_response = requests.get(linked_url)
        this_log.info(f'response of link {linked_response}')

        linked_soup = BeautifulSoup(linked_response.content, 'html.parser')

        # Find and process tables in the linked website
        tables = linked_soup.find_all('table')
        this_log.info(f'find table {len(tables)}')
        for n_table, table in enumerate(tables):
            file_path = p_out + head + '.' + str(n_table) + '.csv'
            content = ''
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                content_row = ','.join([_.get_text() for _ in cells])
                content += content_row+'\n'
            with open(file_path, 'w') as f:
                f.write(content)
            this_log.info(f'save table to {file_path}')

    with open(p_update_flag, 'w') as f:
        f.write(date)


if __name__ == '__main__':
    main()
