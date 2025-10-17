import json
import csv
def time_to_hours(time_str):
    total_hours = 0.0
    parts = time_str.split()
    for i in range(0, len(parts), 2):
        if parts[i+1] == 'hrs':
            total_hours += int(parts[i])
        elif parts[i+1] == 'mins':
            total_hours += int(parts[i]) / 60.0
    return total_hours

def json_to_csv(json_file, csv_file):
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取需要的字段
    csv_data = []
    for recipe in data:
        basic_info = recipe.get('basic_info', {})
        nutritions = recipe.get('nutritions', {})
        prep_data = recipe.get('prep_data',{})
        rating=basic_info.get('rating')
        calories=nutritions.get('calories')
        protein=nutritions.get('protein')
        carbs=nutritions.get('carbs')
        fat=nutritions.get('fat')
        category=basic_info.get("category")
        total_time=prep_data.get('total_time:')
        if total_time is None:
            continue
        else:
            total_time=time_to_hours(total_time)
        if calories is None:
            continue
        if protein is None:
            continue
        elif 'g' in protein:
            protein=protein.replace('g','')
        if carbs is None:
            continue
        elif 'g' in carbs:
            carbs=carbs.replace('g','')
        if fat is None:
            continue
        elif 'g' in fat:
            fat= fat.replace('g','')
        if rating is None:
            continue
        elif '\n' in rating:
            rating=rating.replace('\n','')
        row = {
            'food_name': basic_info.get('title'),
            "calories": calories,
            'protein': protein,
            'carbs': carbs,
            'fat':fat ,
            'rating': rating,
            'category':category,
            'total_time': total_time
        }
        csv_data.append(row)

    # 写入 CSV 文件
    fieldnames = ['food_name',"calories", 'protein', 'carbs', 'fat',  'rating','category','total_time']
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"数据已成功写入 {csv_file}")


# 使用示例
json_file = '../../data/allrecipes.json'
csv_file = '../../data/food_data.csv'
json_to_csv(json_file, csv_file)