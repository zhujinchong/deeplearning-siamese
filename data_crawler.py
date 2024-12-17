# -*- encoding: utf-8 -*-
"""
@File    :   data_crawler.py.py
@Contact :   zhujinchong@foxmail.com
@Author  :   zhujinchong
@Modify Time      @Version    @Desciption
------------      --------    -----------
2024/12/16 14:37    1.0         None
"""

import sys
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager

if sys.platform.startswith('win'):
    is_win = True
else:
    is_win = False
options = EdgeOptions() if is_win else ChromeOptions()
# options.add_argument('headless')  # 无头模式，不打开浏览器窗口
# 无头模式（headless）参数下，需要设置头
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
options.add_argument(f'user-agent={user_agent}')
options.add_argument('--no-sandbox')  # 关闭沙盒模式（提高性能）
options.add_argument('--ignore-certificate-errors')
options.add_argument('--disable-gpu')  # 禁用gpu，解决一些莫名的问题
options.add_argument('--disable-infobars')  # 禁用浏览器正在被自动化程序控制的提示
options.add_argument('--start-maximized')  # 设置浏览器在启动时最大化窗口

if is_win:
    driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()), options=options)
else:
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

driver.get('https://www.tianyancha.com/login?')

# 等待手动进入指定页面
while True:
    x = input("请登录，登录后按回车：")
    break

driver.get('https://www.tianyancha.com/search?key=%E4%B8%89%E5%8F%AA%E7%BE%8A')
while True:
    x = input("打开验证码页面，后按回车：")
    break
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, '.geetest_title .geetest_text_tips'))
)
# 或进入验证码页面
# try:
#     WebDriverWait(driver, 5).until(
#         EC.presence_of_element_located((By.CSS_SELECTOR, '.geetest_ques_tips'))
#     )
# except Exception as e:
#     print("Error: 找不到元素!")
#     exit()

i = 0
while True:
    # 找到title图
    tips = driver.find_element(By.CSS_SELECTOR, '.geetest_ques_tips')
    tips = tips.find_elements(By.TAG_NAME, 'img')  # 保存成图片
    for j, tip in enumerate(tips):
        # tip_url = tip.get_attribute('src')
        image_name = f"./datasets/icons_craw/{str(i)}_tip{str(j)}.png"
        tip.screenshot(image_name)
        print(image_name)

    # 找到背景图
    geetest_bg = driver.find_element(By.CSS_SELECTOR, '.geetest_bg')
    # 保存成图片
    image_name = f"./datasets/icons_craw/{str(i)}_bg.png"
    geetest_bg.screenshot(image_name)
    print(image_name)
    i += 1
    # 刷新按钮
    refresh_btn = driver.find_element(By.CSS_SELECTOR, '.geetest_refresh')
    refresh_btn.click()
    time.sleep(2)
    x = input("按回车键继续：")
