import concurrent.futures
import json
import re
from datetime import datetime
from pathlib import Path

import arrow
import numpy
import scipy
from jsonschema.validators import validate


def historical(filename: Path) -> numpy.ndarray:
    data = []
    with filename.open('r') as f:
        json_data = json.load(f)
    for item in json_data:
        for hist in item['data']:
            data.append([
                arrow.get(hist['point_time']).int_timestamp,
                hist['value'],
            ])

    return numpy.array(data)


def forecast(filename: Path) -> numpy.ndarray:
    data = []
    with filename.open('r') as f:
        json_data = json.load(f)
    for item in json_data:
        for fore in item['data']:
            generated_at = arrow.get(fore['generated_at']).int_timestamp
            d = []
            for data_point in fore['forecast']:
                d.append([generated_at, arrow.get(data_point['point_time']).int_timestamp, data_point['value']])
            d.sort(key=lambda x: x[1])
            data.append(d)
    data.sort(key=lambda x: x[0][0])

    return numpy.array(data)


def main():
    base_path = Path('./dataset')

    historical_data = {}

    for filename in base_path.glob('hist_co2_moer_*.json'):
        region = re.search(r'hist_co2_moer_(.*)_\d+_\d+', filename.stem).group(1)

        historical_data[region] = historical(filename)

    numpy.savez_compressed('historical_data.npz', **historical_data)

    forecast_data = {}

    futures = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for filename in base_path.glob('fore_co2_moer_*.json'):
            region = re.search(r'fore_co2_moer_(.*)_\d+_\d+', filename.stem).group(1)
            futures[region] = executor.submit(
                forecast,
                filename
            )

        for region, future in futures.items():
            forecast_data[region] = future.result()
            print(f'{region} is done!')

    numpy.savez_compressed('forecast_data.npz', **forecast_data)


if __name__ == '__main__':
    main()
