import requests


import requests

def markdown_to_notion(description: str):
    """
    Converts simple Markdown text to a list of Notion blocks.
    Supports: Bold, Italic, and Bullet lists.
    """
    lines = description.split('\n')
    blocks = []
    for line in lines:
        if line.startswith('* '):  # Simple bullet list item
            content = line[2:]
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "text": [{"type": "text", "text": {"content": content}}]
                }
            })
        elif '**' in line:  # Bold text
            start = line.index('**')
            end = line.index('**', start + 2)
            bold_text = line[start+2:end]
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "text": [
                        {"type": "text", "text": {"content": line[:start]}},
                        {"type": "text", "text": {"content": bold_text, "bold": True}},
                        {"type": "text", "text": {"content": line[end+2:]}}
                    ]
                },
                "annotations": {
                    "bold": True,
                    "italic": False,
                    "strikethrough": False,
                    "underline": False,
                    "code": False,
                    "color": "default"
                },

            })
        elif '*' in line:  # Italic text
            start = line.index('*')
            end = line.index('*', start + 1)
            italic_text = line[start+1:end]
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "text": [
                        {"type": "text", "text": {"content": line[:start]}},
                        {"type": "text", "text": {"content": italic_text, "italic": True}},
                        {"type": "text", "text": {"content": line[end+1:]}}
                    ]
                }
            })
        else:  # Regular text
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "text": [{"type": "text", "text": {"content": line}}]
                }
            })
    return blocks

def createEvent(title: str, start_date: str, end_date: str, description: str, icon: str = "📅"):
    token = 'secret_eP7iwZ8idGIgmGf1pQy8nHTZT0u6dofiLVe09ER0KIo'
    database_id = "c98d8f968d734896ab6b6a95628612c0"
    url = f"https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Notion-Version": "2021-05-13"
    }
    notion_blocks = markdown_to_notion(description)
    new_event_data = {
        "parent": {"database_id": database_id},
        "icon": {"type": "emoji", "emoji": icon},
        "properties": {
            "Name": {"title": [{"text": {"content": title}}]},
            "Date": {"type": "date", "date": {"start": start_date, "end": end_date}}
        },
        "children": notion_blocks
    }
    response = requests.post(url, headers=headers, json=new_event_data)
    if response.status_code == 200:
        print("Event added successfully!")
    else:
        print("Failed to add event:", response.text)
    return response
