import datetime
import schedule
import time
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

SLACK_BOT_TOKEN = "xoxb-5114084694709-5110442676198-WKP4EfRUo0Jcb1wdkXZagpkd"
CHANNEL_ID = "C053F0X63M0"
MESSAGE = "30분이 지났습니다. 목과 허리를 곧게 펴고 스트레칭하세요."

def send_message():
    client = WebClient(token=SLACK_BOT_TOKEN)
    try:
        dayOfWeek = datetime.datetime.today().weekday()
        if dayOfWeek < 5: # 월요일부터 금요일까지만 작동
            now = datetime.datetime.now()
            start_time = now.replace(hour=10, minute=0, second=0, microsecond=0)
            end_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
            if start_time <= now <= end_time:
                response = client.chat_postMessage(channel=CHANNEL_ID, text=MESSAGE)
                print("Message sent: ", response["ts"])
                return 
    except SlackApiError as e:
        print("Error sending message: {}".format(e))

schedule.every(30).minutes.do(send_message)

while True:
    schedule.run_pending()
    time.sleep(1)