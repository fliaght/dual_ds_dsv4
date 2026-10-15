#!/usr/bin/env python3
"""Real-browser check of the localStorage round-trip (needs Firefox + geckodriver).

    chat/.venv/bin/python chat/tests/browser_check.py [http://127.0.0.1:9450]

1. open the UI, send a prompt, wait for the reply
2. reload the page -> the chat must come back from localStorage
3. open a fresh private profile -> it must NOT see that chat (isolation)
4. clear -> reload -> gone
Exit 0 on success.
"""
import sys, time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9450"
PROMPT = "What is 17 * 23? Reply with just the number."
GECKO = "/snap/bin/geckodriver"


def driver():
    o = Options(); o.add_argument("-headless")
    return webdriver.Firefox(options=o, service=Service(executable_path=GECKO))


def wait_for(d, cond, timeout=90, every=0.5):
    end = time.time() + timeout
    while time.time() < end:
        try:
            r = cond(d)
            if r: return r
        except Exception:  # noqa: BLE001
            pass
        time.sleep(every)
    raise TimeoutError("condition not met")


def body_text(d): return d.find_element(By.TAG_NAME, "body").text
def chat_box(d): return d.find_element(By.CSS_SELECTOR, "textarea[data-testid='stChatInputTextArea']")
def ls(d, key): return d.execute_script("return window.localStorage.getItem(arguments[0]);", key)


def send(d, text):
    box = wait_for(d, chat_box, 60)
    box.send_keys(text); box.send_keys(Keys.ENTER)


def main():
    ok = True
    d = driver()
    try:
        d.get(URL)
        wait_for(d, chat_box, 60)
        send(d, PROMPT)
        wait_for(d, lambda x: "391" in body_text(x), 120)
        wait_for(d, lambda x: ls(x, "dsv4chat:conversation") and "391" in ls(x, "dsv4chat:conversation"), 20)
        cid = ls(d, "dsv4chat:client_id")
        print(f"[1] reply received and saved to localStorage (client_id={cid[:8]}…)")

        d.refresh()
        wait_for(d, chat_box, 60)
        wait_for(d, lambda x: "391" in body_text(x) and PROMPT in body_text(x), 30)
        print("[2] reload -> chat restored from localStorage")

        d2 = driver()   # separate profile = another visitor's browser
        try:
            d2.get(URL); wait_for(d2, chat_box, 60); time.sleep(2)
            if "391" in body_text(d2) or ls(d2, "dsv4chat:conversation"):
                print("[3] FAIL: another browser can see the chat"); ok = False
            else:
                print("[3] another browser sees nothing (isolated)")
        finally:
            d2.quit()

        d.find_element(By.XPATH, "//button[contains(., 'Clear chat')]").click()
        wait_for(d, lambda x: ls(x, "dsv4chat:conversation") is None, 20)
        d.refresh(); wait_for(d, chat_box, 60); time.sleep(2)
        if "391" in body_text(d):
            print("[4] FAIL: chat still present after clear + reload"); ok = False
        else:
            print("[4] clear + reload -> gone")
    finally:
        d.quit()
    print("BROWSER CHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
