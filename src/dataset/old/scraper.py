from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.wait import WebDriverWait
from datetime import date, timedelta
from utils import full_url
import time
import re
import json
import click
import gc
import sys
from multiprocessing import Pool


class Scraper:
    def __init__(self, headless):
        options = webdriver.FirefoxOptions()
        if headless:
            options.add_argument("--headless")
        options.set_preference("pageLoadStrategy", "eager")

        profile = webdriver.FirefoxProfile()
        profile.set_preference("permissions.default.image", 2)
        profile.set_preference("permissions.default.stylesheet", 2)
        profile.set_preference("network.cookie.cookieBehavior", 2)

        options.profile = profile

        self.driver = webdriver.Firefox(options=options)
        self.driver.set_page_load_timeout(180)

    def __del__(self):
        self.driver.quit()

    def game_hrefs_from_date(self, useddate, retry=False):
        try:
            self.driver.get(full_url(useddate))
        except:
            if retry:
                raise
            return self.game_hrefs_from_date(useddate, True)

        # nav to main content, summary page, query within
        index = self.driver.find_element(By.CLASS_NAME, "index")
        try:
            summaries = index.find_element(By.CLASS_NAME, "game_summaries")
        except:
            return []

        links = summaries.find_elements(By.CLASS_NAME, "right.gamelink")

        # process result
        hrefs = [
            link.find_element(By.TAG_NAME, "a").get_attribute("href") for link in links
        ]

        return hrefs

    def game_data_from_href(self, href, date, retry=False):
        try:
            self.driver.get(href + "#four_factors")
        except:
            if retry:
                raise
            return self.game_data_from_href(href, date, True)
        res = dict()

        # ----- find h1 button and click -----

        try:
            filter_switcher = self.driver.find_element(By.CLASS_NAME, "filter.switcher")
            filter_switcher.find_elements(By.TAG_NAME, "div")[3].click()
        except Exception as e:
            if retry:
                raise
            return self.game_data_from_href(href, date, True)
        # ActionChains(self.driver).move_to_element(button).click(button).perform()

        # ----- find stats tables -----
        # content = self.driver.find_element(By.ID, "content")
        try:
            foots = [
                foot
                for foot in self.driver.find_elements(By.TAG_NAME, "tfoot")
                if foot.is_displayed()
            ]
        except Exception as e:
            if retry:
                raise
            return self.game_data_from_href(href, date, True)

        if len(foots) != 2:
            return self.game_data_from_href(href, date, False)

        res["away_stats"] = dict()
        for stat in foots[0].find_elements(By.TAG_NAME, "td"):
            res["away_stats"][stat.get_attribute("data-stat")] = stat.get_attribute(
                "innerHTML"
            )

        res["home_stats"] = dict()
        for stat in foots[1].find_elements(By.TAG_NAME, "td"):
            res["home_stats"][stat.get_attribute("data-stat")] = stat.get_attribute(
                "innerHTML"
            )

        # ----- referees -----
        # content = self.driver.find_element(By.ID, "content")
        officials_element = [
            item
            for item in self.driver.find_elements(By.TAG_NAME, "strong")
            if "Officials:" in item.get_attribute("innerHTML")
        ][0]
        all_officials = [
            item.get_attribute("href")
            for item in officials_element.find_elements(
                By.XPATH, "./following-sibling::a"
            )
        ]
        res["officials"] = [
            re.search(r"\/(\w+?)\.html$", off).group(1) for off in all_officials
        ]

        ### unused ###
        """
        #----- attendance -----
        att_element = [item for item in content.find_elements(By.TAG_NAME, "strong") if "Attendance:" in item.get_attribute("innerHTML")][0]
        att_inner = att_element.find_element(By.XPATH, "..").get_attribute("innerHTML")
        attendance = att_inner.split(">")[-1]
        res["attendance"] = int(attendance.replace(",", ""))

        #----- time of game -----
        tog_element = [item for item in content.find_elements(By.TAG_NAME, "strong") if "Time of Game:" in item.get_attribute("innerHTML")][0]
        tog_inner = tog_element.find_element(By.XPATH, "..").get_attribute("innerHTML")
        time_of_game = tog_inner.split(">")[-1]
        res["time_of_game"] = time_of_game
        """

        # ----- team names -----
        sbox = self.driver.find_element(By.CLASS_NAME, "scorebox")
        link_divs = sbox.find_elements(By.TAG_NAME, "strong")
        link_hrefs = [
            l.find_element(By.TAG_NAME, "a").get_attribute("href") for l in link_divs
        ]
        res["away"] = link_hrefs[0].split("/")[-2]
        res["home"] = link_hrefs[1].split("/")[-2]

        # ----- end score -----
        scores = sbox.find_elements(By.CLASS_NAME, "score")
        res["final_score"] = [int(s.get_attribute("innerHTML")) for s in scores]

        return res

    def run_date(self, date):
        hrefs = self.game_hrefs_from_date(date)

        for i, link in enumerate(hrefs):
            data = self.game_data_from_href(link, date)
            if data:
                with open(
                    f"data/{date.strftime('%Y-%m-%d')}-{data["away"]}-{data["home"]}.json",
                    "w",
                ) as file:
                    file.write(json.dumps(data, indent=4))


def date_range(end, diff, delta=1):
    start = end - timedelta(days=diff)
    while start <= end:
        yield start
        start += timedelta(days=delta)


def run_single_date(date):
    global hl
    scraper = Scraper(hl)
    scraper.run_date(date)
    del scraper


def run_range(n_days, batch_size):
    with Pool(batch_size) as p:
        p.map(run_single_date, date_range(date.today(), n_days))


@click.command()
@click.option("--days", required=True, help="number of days before today to check")
@click.option("--batch-size", required=True, help="Number of days run at a time")
@click.option(
    "--headless/--no-headless", default=False, help="headless disables browser ui"
)
def main(days, batch_size, headless):
    """Generates JSON data files for requested dates"""
    global hl
    hl = headless

    run_range(int(days), int(batch_size))


if __name__ == "__main__":
    main()
