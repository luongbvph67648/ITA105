from collections import Counter
from datetime import datetime
from pathlib import Path
import argparse
import numpy as np # Added import for numpy
import csv
import statistics


def parse_bool(value: str) -> bool:
    return value.strip() in {"1", "True", "true", "yes", "Y", "y"}


def parse_date(value: str) -> datetime.date:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def load_data(csv_path: Path):
    data = []
    with csv_path.open(newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        # Helper function to safely convert to float
        def safe_float(value):
            try:
                return float(value)
            except ValueError:
                return np.nan

        # Helper function to safely convert to int
        def safe_int(value):
            try:
                return int(value)
            except ValueError:
                return np.nan

        for row in reader:
            data.append({ # Use safe converters for numeric fields
                "LotArea": safe_float(row["LotArea"]),
                "SalePrice": safe_float(row["SalePrice"]),
                "Rooms": safe_int(row["Rooms"]),
                "HasGarage": parse_bool(row["HasGarage"]),
                "NoiseFeature": safe_float(row["NoiseFeature"]),
                "Neighborhood": row["Neighborhood"].strip(),
                "Condition": row["Condition"].strip(),
                "Description": row["Description"].strip(),
                "SaleDate": parse_date(row["SaleDate"]),
                "ImagePath": row["ImagePath"].strip(),
            })
    return data


# Moved compute_simple_lot_area_model outside of summarize
def compute_simple_lot_area_model(data):
    # Filter out rows where LotArea or SalePrice is NaN
    filtered_data = [(row["LotArea"], row["SalePrice"]) for row in data if not np.isnan(row["LotArea"]) and not np.isnan(row["SalePrice"])]

    if len(filtered_data) < 2:
        return None, None

    x = [item[0] for item in filtered_data]
    y = [item[1] for item in filtered_data]

    n = len(x)
    x_mean = np.mean(x) # Use numpy mean now that we've filtered NaNs
    y_mean = np.mean(y) # Use numpy mean

    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)
    if denominator == 0:
        return None, None

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def summarize(data):
    sale_prices = [row["SalePrice"] for row in data]
    lot_areas = [row["LotArea"] for row in data]
    rooms = [row["Rooms"] for row in data]
    garages = [row["HasGarage"] for row in data]

    print("Dataset summary")
    print("--------------")
    print(f"Total records: {len(data)}")
    print(f"Average sale price: {np.nanmean(sale_prices):,.2f}") # Use np.nanmean
    print(f"Median sale price: {np.nanmedian(sale_prices):,.2f}") # Use np.nanmedian
    print(f"Average lot area: {np.nanmean(lot_areas):,.2f}") # Use np.nanmean
    print(f"Average number of rooms: {np.nanmean(rooms):.2f}") # Use np.nanmean
    print(f"Households with garage: {sum(garages)} / {len(data)} ({sum(garages) / len(data) * 100:.1f}%)")
    print()

    neighborhood_counts = Counter(row["Neighborhood"] for row in data)
    condition_counts = Counter(row["Condition"] for row in data)
    print("Neighborhood distribution")
    for neighborhood, count in neighborhood_counts.most_common():
        print(f"  {neighborhood}: {count}")
    print()
    print("Condition distribution")
    for condition, count in condition_counts.most_common():
        print(f"  {condition}: {count}")
    print()

    best = max(data, key=lambda row: row["SalePrice"])
    worst = min(data, key=lambda row: row["SalePrice"])
    print("Most expensive house")
    print(f"  Price: {best['SalePrice']:.2f}, Rooms: {best['Rooms']}, Neighborhood: {best['Neighborhood']}, Condition: {best['Condition']}")
    print(f"  Description: {best['Description']}")
    print()
    print("Least expensive house")
    print(f"  Price: {worst['SalePrice']:.2f}, Rooms: {worst['Rooms']}, Neighborhood: {worst['Neighborhood']}, Condition: {worst['Condition']}")
    print(f"  Description: {worst['Description']}")
    print()

def print_prediction_model(data):
    slope, intercept = compute_simple_lot_area_model(data)
    if slope is None:
        print("Không thể xây dựng mô hình dự đoán với dữ liệu hiện tại.")
        return

    print("Simple linear model: SalePrice = intercept + slope * LotArea")
    print(f"  intercept = {intercept:.2f}")
    print(f"  slope = {slope:.4f}") # Use np.nanmean for sample_area
    sample_area = np.nanmean([row["LotArea"] for row in data])
    predicted = intercept + slope * sample_area
    print(f"  Example prediction for average lot area ({sample_area:.2f}): {predicted:.2f}")
    print()


def find_house_by_keyword(data, keyword: str):
    keyword_lower = keyword.lower()
    results = [row for row in data if keyword_lower in row["Description"].lower()]
    print(f"Houses with keyword '{keyword}' in description: {len(results)}")
    for row in results[:5]:
        print(f"  Price={row['SalePrice']:.2f}, Rooms={row['Rooms']}, Neighborhood={row['Neighborhood']}, Description={row['Description']}")
    print()


def print_first_rows(data, count: int = 5):
    print(f"First {count} rows")
    print("----------------")
    for row in data[:count]:
        print(f"{row['SaleDate']} | {row['Neighborhood']} | Price={row['SalePrice']:.2f} | Rooms={row['Rooms']} | Garage={row['HasGarage']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Chạy Lab 8 với tập dữ liệu 'ITA105_Lab_8.csv'.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).with_name("ITA105_Lab_8.csv"),
        help="Đường dẫn tới tệp CSV (mặc định: ITA105_Lab_8.csv).",
    )
    parser.add_argument("--keyword", help="Tìm kiếm nhà theo từ khóa trong mô tả.")
    args = parser.parse_args()

    csv_path = args.file
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp CSV: {csv_path}")

    data = load_data(csv_path)
    if not data:
        print("Tệp CSV rỗng hoặc không có dữ liệu hợp lệ.")
        return

    print_first_rows(data)
    summarize(data)
    print_prediction_model(data)
    if args.keyword:
        find_house_by_keyword(data, args.keyword)

if __name__ == "__main__":
    main()