import re
import wordninja

def is_en_or(text):
    english_pattern = re.compile(r'[a-zA-Z]')
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')

    english_count = len(english_pattern.findall(text))
    chinese_count = len(chinese_pattern.findall(text))

    if english_count > chinese_count:
        return True
    else:
        return False

def tokenise_english_query(text):
    split_text = wordninja.split(text)
    print("word", split_text)
    return " ".join(split_text)