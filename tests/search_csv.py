"""在csv文件中搜索包含特定关键词的行，并返回这些行的内容"""

import csv
from pathlib import Path

def search_csv(file_path: Path, keyword: str) -> list[dict[str, str]]:
    results = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if keyword in ''.join(row.values()):
                results.append(row)
    return results

if __name__ == "__main__":
    csv_file = Path("datasets\\lmsys-chat-lewd-filter.csv")  # 替换为你的csv文件路径
    search_keyword = "sheep"  # 替换为你要搜索的关键词
    matching_rows = search_csv(csv_file, search_keyword)
    for row in matching_rows:
        print(row)