
from scrapy.crawler import CrawlerProcess
from spiders.search import SearchSpider
from scrapy.utils.project import get_project_settings

if __name__ == "__main__":
    process = CrawlerProcess(get_project_settings())
    process.crawl(SearchSpider)
    process.start()  # The script will block here until the crawling is finished
#
# def main(keyword):
#     """Main function to run the Weibo spider with a keyword."""
#     # Set up the process
#     process = CrawlerProcess({
#         'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#         'DEFAULT_REQUEST_HEADERS': {
#             'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
#             'Accept-Language': 'en'
#         },
#         # Additional Scrapy settings
#         'LOG_LEVEL': 'INFO',
#         # Enable/Disable other Scrapy features here, such as 'FEEDS' to store output
#     })
#
#     # Start the spider
#     # process.crawl(WeiboSpider, keyword=keyword)
#     process.crawl(SearchSpider)
#     process.start()
#
#
# if __name__ == "__main__":
#     # Call the main function with the desired keyword
#     keyword = "Artificial Intelligence"  # Example search keyword
#     main(keyword)
