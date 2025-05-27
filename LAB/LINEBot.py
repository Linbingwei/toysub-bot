from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from RAG import get_rag_answer
from linebot.models import ImageSendMessage, TextSendMessage
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, FlexSendMessage,
    QuickReply, QuickReplyButton, MessageAction
)
import requests

app = Flask(__name__)

CHANNEL_ACCESS_TOKEN = ''
CHANNEL_SECRET = ''

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)
user_state = {}

#啟動 loading indicator
def start_loading_indicator(user_id, seconds=30):
    url = "https://api.line.me/v2/bot/chat/loading/start"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}"
    }
    payload = {
        "chatId": user_id,
        "loadingSeconds": seconds 
    }
    response = requests.post(url, headers=headers, json=payload)
    print("啟動 loading indicator 回應:", response.status_code, response.text)


@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_msg = event.message.text.strip()
    user_id = event.source.user_id
    print("使用者輸入內容：", user_msg)

    # ==== 固定功能：使用者回饋 ====
    if user_msg == "優惠方案":
        bubble = {
            "type": "bubble",
            "hero": {
                "type": "image",
                "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRpJBKQMkVKtSfp1Cq4vcXFDWaFX2_JC-Acng&s",
                "size": "full",
                "aspectRatio": "1:1",
                "aspectMode": "cover",
                "action": {
                    "type": "uri",
                    "uri": "https://toysub.tw"
                }
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "優惠方案",
                        "weight": "bold",
                        "size": "md",
                        "wrap": True
                    }
                ]
            }
        }
        flex_msg = FlexSendMessage(alt_text="優惠方案", contents=bubble)
        line_bot_api.reply_message(event.reply_token, flex_msg)
        return

    # ==== 固定功能：診斷問答 ====
    elif user_msg == "新品介紹" or user_msg == "新品":
        user_state[user_id] = {"step": 1}
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text="讓我們幫你找出最適合寶寶的玩具🎉\n請問孩子是？",
                quick_reply=QuickReply(items=[
                    QuickReplyButton(action=MessageAction(label="男孩", text="男孩")),
                    QuickReplyButton(action=MessageAction(label="女孩", text="女孩"))
                ])
            )
        )
        return

    if user_id in user_state and user_state[user_id].get("step"):
        step = user_state[user_id]["step"]

        if step == 1:
            user_state[user_id]["gender"] = user_msg
            user_state[user_id]["step"] = 2
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text="請問孩子目前的年齡是？",
                    quick_reply=QuickReply(items=[
                        QuickReplyButton(action=MessageAction(label="3-6 個月", text="3-6 個月")),
                        QuickReplyButton(action=MessageAction(label="7-11 個月", text="7-11 個月")),
                        QuickReplyButton(action=MessageAction(label="1 歲", text="1 歲")),
                        QuickReplyButton(action=MessageAction(label="2 歲", text="2 歲")),
                        QuickReplyButton(action=MessageAction(label="3 歲", text="3 歲")),
                        QuickReplyButton(action=MessageAction(label="4-6 歲", text="4-6 歲"))
                    ])
                )
            )
            return

        elif step == 2:
            user_state[user_id]["age"] = user_msg
            user_state[user_id]["step"] = 3
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text="孩子對哪類玩具特別感興趣？",
                    quick_reply=QuickReply(items=[
                        QuickReplyButton(action=MessageAction(label="手部操作型", text="手部操作型")),
                        QuickReplyButton(action=MessageAction(label="體能活動型", text="體能活動型")),
                        QuickReplyButton(action=MessageAction(label="益智學習型", text="益智學習型"))
                    ])
                )
            )
            return

        elif step == 3:
            user_state[user_id]["interest"] = user_msg
            user_state[user_id]["step"] = 4
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text="是否有特別需求？",
                    quick_reply=QuickReply(items=[
                        QuickReplyButton(action=MessageAction(label="安靜無聲", text="安靜無聲")),
                        QuickReplyButton(action=MessageAction(label="可水洗", text="可水洗")),
                        QuickReplyButton(action=MessageAction(label="沒特別需求", text="沒特別需求"))
                    ])
                )
            )
            return

        elif step == 4:
            user_state[user_id]["special"] = user_msg
            gender = user_state[user_id]["gender"]
            age = user_state[user_id]["age"]
            interest = user_state[user_id]["interest"]
            special = user_state[user_id]["special"]
            result_text = f"🎉 根據診斷結果：\n性別：{gender}\n年齡：{age}\n興趣：{interest}\n需求：{special}\n\n請至我們官網查看最適合的玩具 👉 https://toysub.tw"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=result_text))
            del user_state[user_id]
            return

    # ==== 開啟智能客服模式 ====
    if user_msg == "客服機器人":
        user_state[user_id] = {"mode": "rag"}
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="您好，請問有什麼需要幫忙？請直接輸入您的問題。")
        )
        return

    # ==== 智能客服回答 ====
    if user_state.get(user_id, {}).get("mode") == "rag":
        try:
            start_loading_indicator(user_id, seconds=60)
            reply = get_rag_answer(user_msg)
            line_bot_api.push_message(
                user_id,
                TextSendMessage(text=reply)
            )

        except Exception as e:
            line_bot_api.push_message(
                user_id,
                TextSendMessage(text=f"客服系統暫時無法使用，請稍後再試 🙇\n錯誤訊息：{e}")
            )
        return

    # ==== 其他一般訊息 ====
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="感謝您的訊息，我們會盡快回覆。")
    )

if __name__ == "__main__":
    app.run(port=5000)