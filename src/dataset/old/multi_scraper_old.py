from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from datetime import date, timedelta
from joblib import Parallel, delayed
from utils import full_url
from threading import Lock
import time
import re
import json
import click
import gc
import sys

progress = {"len": 0, "tasks": dict()}
task_lock = Lock()


class Scraper:
    def __init__(self, headless):
        options = webdriver.FirefoxOptions()
        if headless:
            options.add_argument("--headless")
        options.set_preference("pageLoadStrategy", "eager")

        profile = webdriver.FirefoxProfile()
        profile.set_preference("permissions.default.image", 2)
        profile.set_preference("permissions.default.stylesheet", 2)

        options.profile = profile

        self.driver = webdriver.Firefox(options=options)
        self.driver.set_page_load_timeout(180)

    def __del__(self):
        self.driver.quit()

    def game_hrefs_from_date(self, useddate, retry=False):
        global task_lock
        with task_lock:
            global progress
            progress["tasks"][useddate] = [0, 0, 0]

        update_progress()

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
            with task_lock:
                progress["tasks"][useddate] = [0, 0, 2]
            update_progress()
            return []

        links = summaries.find_elements(By.CLASS_NAME, "right.gamelink")

        # process result
        hrefs = [
            link.find_element(By.TAG_NAME, "a").get_attribute("href") for link in links
        ]

        with task_lock:
            progress["tasks"][useddate] = [0, len(hrefs), 1]
        update_progress()

        return hrefs

    def game_data_from_href(self, href, date, retry=False):
        try:
            self.driver.get(href)
        except:
            if retry:
                raise
            return self.game_data_from_href(href, date, True)
        res = dict()

        # ----- find h1 button and click -----
        filter_switcher = self.driver.find_element(By.CLASS_NAME, "filter.switcher")
        button = filter_switcher.find_elements(By.TAG_NAME, "div")[3]
        self.scroll_shim(button)
        time.sleep(1)
        ActionChains(self.driver).click(button).perform()

        # ----- find stats tables -----
        content = self.driver.find_element(By.ID, "content")
        foots = [
            foot
            for foot in content.find_elements(By.TAG_NAME, "tfoot")
            if foot.is_displayed()
        ]

        if len(foots) != 2:
            return self.game_data_from_href(href, date, retry)

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
        content = self.driver.find_element(By.ID, "content")
        officials_element = [
            item
            for item in content.find_elements(By.TAG_NAME, "strong")
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
        sbox = content.find_element(By.CLASS_NAME, "scorebox")
        link_divs = sbox.find_elements(By.TAG_NAME, "strong")
        link_hrefs = [
            l.find_element(By.TAG_NAME, "a").get_attribute("href") for l in link_divs
        ]
        res["away"] = link_hrefs[0].split("/")[-2]
        res["home"] = link_hrefs[1].split("/")[-2]

        # ----- end score -----
        scores = sbox.find_elements(By.CLASS_NAME, "score")
        res["final_score"] = [int(s.get_attribute("innerHTML")) for s in scores]
        global task_lock
        with task_lock:
            global progress
            progress["tasks"][date][0] += 1
        update_progress()

        return res

    def scroll_shim(self, obj):
        y = obj.location["y"]
        self.driver.execute_script(
            f"window.scrollTo({{top: {max(y - 120, 0)}, behavior: 'smooth'}});"
        )

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
        with task_lock:
            global progress
            progress["tasks"][date][2] = 2
        update_progress()


def update_progress():
    global task_lock
    with task_lock:
        global progress
        sys.stdout.write(f"\033[{progress["len"]}A")
        for date, [start, end, finished] in progress["tasks"].items():
            match finished:
                case 0:
                    sys.stdout.write(
                        f"\r{date}- [{"#" * start}{"-" * (end - start)}] {start} / {end}\033[0m\033[K\n"
                    )
                case 1:
                    sys.stdout.write(
                        f"\r\033[31m{date}- [{"#" * start}{"-" * (end - start)}] {start} / {end}\033[0m\033[K\n"
                    )
                case 2:
                    sys.stdout.write(
                        f"\r\033[32m{date}- [{"#" * start}{"-" * (end - start)}] {start} / {end}\033[0m\033[K\n"
                    )

        sys.stdout.flush()
        progress["len"] = len(progress["tasks"])


def date_range(end, diff):
    diff = timedelta(days=diff)
    start = end - diff
    while start < end:
        yield start
        start += timedelta(days=1)


def run_single_date(date, headless):
    scraper = Scraper(headless)
    scraper.run_date(date)
    del scraper
    gc.collect()


def run_range(n_days, headless):
    Parallel(n_jobs=2, backend="threading", timeout=300)(
        delayed(run_single_date)(date, headless)
        for date in date_range(date.today(), n_days)
    )


@click.command()
@click.option("--days", required=True, help="number of days before today to check")
@click.option(
    "--headless/--no-headless", default=False, help="headless disables browser ui"
)
def main(days, headless):
    """Generates JSON data files for requested dates"""

    run_range(int(days), headless)


if __name__ == "__main__":
    main()
