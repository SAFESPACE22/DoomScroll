# autoScroll.py

import pyautogui

def scroll_function():
    speed = -100
    sleep_time = 1  # Set sleep time in seconds
    pyautogui.time.sleep(sleep_time)  # Wait for 3 seconds before scrolling
    pyautogui.scroll(int(speed))  # Scroll with the specified speed
    pyautogui.time.sleep(1)  # Adjust this sleep time as needed after scrolling
