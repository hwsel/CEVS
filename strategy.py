import concurrent.futures
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy

type CorrectionType = Callable[[int | float], int | float] | str
type RegionType = str | int
type TimestampType = int


class Strategy(object):

    def __init__(
            self,
            datacenters: int | list[str],
            setup_time: int = 0,
            hold_time: int = 0,
            coexist_time: int | None = None,
            correction: list[CorrectionType] | dict[str, CorrectionType] | None = None,
    ):
        if isinstance(datacenters, int):
            self.named = False
            self.data = {idx: [] for idx in range(datacenters)}  # type: dict[RegionType, list[float]]
            self.forecast = {idx: [] for idx in range(datacenters)}  # type: dict[RegionType, list[float]]
            self.size = datacenters

            if not isinstance(correction, list) and correction is not None:
                raise TypeError("Correction must be a list or None")
            if isinstance(correction, list) and len(correction) != self.size:
                raise ValueError("Correction size must be equal to datacenters")
            if correction is None:
                self.correction = {idx: lambda x: x for idx in
                                   range(datacenters)}  # type: dict[RegionType, CorrectionType]
            else:
                self.correction = {idx: correction[idx] if callable(correction[idx]) else eval(correction[idx]) for idx
                                   in range(datacenters)}  # type: dict[RegionType, CorrectionType]


        elif isinstance(datacenters, list) and isinstance(datacenters[0], str):
            self.named = True
            self.data = {idx: [] for idx in datacenters}  # type: dict[RegionType, list[float]]
            self.forecast = {idx: [] for idx in datacenters}  # type: dict[RegionType, list[float]]
            self.size = len(datacenters)

            if not isinstance(correction, dict) and correction is not None:
                raise TypeError("Correction must be a dict or None")
            if correction is None:
                self.correction = {idx: lambda x: x for idx in datacenters}  # type: dict[RegionType, CorrectionType]
            else:
                self.correction = {}  # type: dict[RegionType, CorrectionType]
                for idx in datacenters:
                    if idx not in correction:
                        raise ValueError(f'region {idx} not in correction')
                    self.correction[idx] = correction[idx] if callable(correction[idx]) else eval(correction[idx])
        else:
            raise TypeError('datacenters must be int or list of str')

        self.setup_time = setup_time
        self.hold_time = hold_time
        self.coexist_time = coexist_time

        self.sampling_interval = 5 * 60
        self.current_timestamp: int = -1
        self.residual = float('inf')

        self.no_change_until = -1
        self.accumulate_from = None  # type: tuple[TimestampType, RegionType, float, int] | None  # (timestamp, region, reading, rest time)
        self.candidates = []  # type: list[tuple[TimestampType, RegionType]]  # [(timestamp, region)]

        self.history = []  # type: list[tuple[TimestampType, RegionType, float]]  # [(timestamp, region, moer)]

    def _update_validate(self, data):
        """Validate the input of the function `update()`

        Only accepts list of floats, or dict of str-float pairs.

        :param data:
        :return:
        """
        if not (isinstance(data, dict) or isinstance(data, list)):
            raise TypeError('data must be a dict or list')

        if len(data) != self.size:
            raise TypeError('data must have same size as datacenters')

        if isinstance(data, list) and not all(isinstance(v, float) for v in data):
            raise TypeError('data must be a list of floats')

        if isinstance(data, dict) and not (
                all(isinstance(k, str) for k in data.keys()) or all(isinstance(v, float) for v in data.values())):
            raise TypeError('data must be a dict of (str: float)')

    def update(
            self,
            measurement: list[float] | dict[str, float],
            prediction: list[float] | dict[str, float] | None = None,
            timestamp: int = -1
    ) -> tuple[TimestampType, RegionType, float]:
        # validate inputs
        self._update_validate(measurement)
        if prediction:
            self._update_validate(prediction)

        # handle timestamp input
        if timestamp == -1:
            if self.current_timestamp == -1:
                timestamp = 0
            else:
                timestamp = self.current_timestamp + self.sampling_interval

        # update measurements
        if self.named:
            for region in self.data:
                if region not in measurement:
                    raise KeyError('region {} not in measurement'.format(region))

                self.data[region].append(measurement[region])
        else:
            for idx in range(self.size):
                self.data[idx].append(measurement[idx])

        # update forecast
        if prediction:
            if self.named:
                for region in self.forecast:
                    if region not in prediction:
                        raise KeyError('region {} not in prediction'.format(region))

                    self.forecast[region].append(prediction[region])
            else:
                for idx in range(self.size):
                    self.forecast[idx].append(prediction[idx])

        # update current timestamp
        self.current_timestamp = timestamp

        # DONE: update predictions

        # determine next one

        # get current (region, value) pair and sort based on the value
        if prediction:
            sorted_current = [(key, value[-1]) for key, value in self.forecast.items()]
        else:
            sorted_current = [(key, value[-1]) for key, value in self.data.items()]

        # DONE: corrections
        sorted_current = [(key, self.correction[key](value)) for key, value in sorted_current]

        sorted_current.sort(key=lambda x: x[1])

        current_minimal = [item for item in sorted_current if
                           item[1] == sorted_current[0][1]]  # type: list[tuple[RegionType, float]]

        # remove all non-min from candidates  # [(timestamp, region)]
        self.candidates = [candidate for candidate in self.candidates if
                           candidate[1] in [region for region, _ in current_minimal]]
        # add new candidates
        for region, _ in current_minimal:
            # if not new
            if region in [candidate[1] for candidate in self.candidates]:
                continue
            self.candidates.append((self.current_timestamp, region))
        # sort candidates, if add at the same time, sort based on region name
        self.candidates.sort(key=lambda x: x[1])
        self.candidates.sort(key=lambda x: x[0])

        # if the using one is still one of the minimals
        last_region = self.history[-1][1] if self.history else None
        last_region_current_measurement = self.data[last_region][-1] if last_region is not None else None

        # handle dynamic
        # (timestamp, region, reading, rest time)
        if self.accumulate_from:

            from_timestamp, from_region, from_measurement, from_coexist_time = self.accumulate_from
            # just accumulate
            if self.current_timestamp - from_timestamp < from_coexist_time:
                self.residual += from_measurement * (self.current_timestamp - from_timestamp)
            elif from_coexist_time > 0:
                self.residual += from_measurement * from_coexist_time
                self.residual -= (from_measurement - last_region_current_measurement) * (
                        self.current_timestamp - from_timestamp - from_coexist_time)
            else:
                self.residual -= (from_measurement - last_region_current_measurement) * (
                        self.current_timestamp - from_timestamp)

            if self.residual > 0:
                self.accumulate_from = (self.current_timestamp, from_region, self.data[from_region][-1],
                                        from_coexist_time - (self.current_timestamp - from_timestamp))
            else:
                self.accumulate_from = None

            # print('ACC: ', from_timestamp, from_region, from_measurement, from_coexist_time, self.residual)

        # initialize
        if not last_region:
            # use the lowest
            new_region = self.candidates[0][1]
            self.history.append(
                (self.current_timestamp, new_region, self.data[new_region][-1]))  # [(timestamp, region, moer)]
            # TODO: does the initial one have the hold time? Currently, YES.
            self.no_change_until = self.current_timestamp + self.hold_time
        # in hold time, no action
        elif self.current_timestamp < self.no_change_until and self.residual > 0:
            # use the last region, and update the new measurement of the last region
            self.history.append((self.current_timestamp, last_region, last_region_current_measurement))
        # should use new candidate                and the candidate meet the setup time
        elif last_region != self.candidates[0][1] and self.current_timestamp - self.candidates[0][0] >= self.setup_time:
            new_region = self.candidates[0][1]
            self.history.append((self.current_timestamp, new_region, self.data[new_region][-1]))
            self.no_change_until = self.current_timestamp + self.hold_time
            if self.coexist_time:
                self.accumulate_from = (self.current_timestamp, last_region, last_region_current_measurement,
                                        self.coexist_time)  # (timestamp, region, reading, rest time)
                self.residual = 0
        else:
            self.history.append((self.current_timestamp, last_region, last_region_current_measurement))

        return self.history[-1]

    def get_history(self):
        return self.history

    def get_carbon(self):
        timestamps = numpy.array([item[0] for item in self.history], dtype=numpy.int_)
        values = numpy.array([item[2] for item in self.history])
        return float(numpy.sum((timestamps[1:] - timestamps[:-1]) * values[:-1]))

    def get_switches(self):
        return sum(1 for idx in range(1, len(self.history)) if self.history[idx][1] != self.history[idx - 1][1])

    @classmethod
    def evaluate(
            cls,
            datacenters: list[str] | None = None,
            setup_time: int = 0,
            hold_time: int = 0,
            coexist_time: int | None = None,
            correction: list[CorrectionType] | dict[str, CorrectionType] | None = None,
            start: int | None = None,
            end: int | None = None,
            predict: int | None = None,
            historical_data: str | Path = 'historical_data.npz',
            forecast_data: str | Path = 'forecast_data.npz',
    ):
        datacenters.sort()

        measurement, forecast = prepare_data(
            regions=datacenters,
            start=start,
            end=end,
            predict=predict,
            historical_data=historical_data,
            forecast_data=forecast_data,
        )

        # print(measurement.shape)

        strategy = cls(
            datacenters=datacenters,  # hack here
            setup_time=setup_time,
            hold_time=hold_time,
            coexist_time=coexist_time,
            correction=correction,
        )

        for idx in range(measurement.shape[0]):
            strategy.update(
                measurement={datacenters[reg]: float(measurement[idx, reg + 1]) for reg in range(len(datacenters))},
                prediction={datacenters[reg]: float(forecast[idx, reg + 1]) for reg in
                            range(len(datacenters))} if forecast is not None else None,
                timestamp=int(measurement[idx, 0])
            )

        return strategy.get_carbon(), strategy.get_switches(), strategy.get_history()


def prepare_data(
        regions: list[str] | None = None,
        start: int | None = None,
        end: int | None = None,
        predict: int | None = None,
        historical_data: str | Path = 'historical_data.npz',
        forecast_data: str | Path = 'forecast_data.npz',
):
    """Get a set or a subset of the dataset

    If pass `regions`, only the selected regions will be returned.

    All regions are sorted. Only the records with common timestamps will be returned.

    Returning is a `numpy.ndarray` array, where

    [
        [timestamp, region0, region1, ..., regionN],
        [timestamp, region0, region1, ..., regionN],
        ...
        [timestamp, region0, region1, ..., regionN],
    ]

    :param predict:
    :param regions:
    :param start:
    :param end:
    :param historical_data:
    :param forecast_data:
    :return:
    """
    if isinstance(historical_data, str):
        historical_data = Path(historical_data)
    if isinstance(forecast_data, str):
        forecast_data = Path(forecast_data)

    historical_npz_data = numpy.load(str(historical_data))
    forecast_npz_data = numpy.load(str(forecast_data))

    if not regions:
        all_regions = [set(historical_npz_data.keys()), ]
        if predict is not None:
            all_regions.append(set(forecast_npz_data.keys()))

        regions = sorted(set.intersection(*all_regions))

    historical_list = [historical_npz_data[region] for region in sorted(regions)]
    if predict is not None:
        forecast_list = [forecast_npz_data[region] for region in sorted(regions)]

    all_timestamps = [set(data[:, 0]) for data in historical_list]
    if predict is not None:
        all_timestamps.extend([set(data[:, 0, 0]) for data in forecast_list])

    common_ts = sorted(set.intersection(*all_timestamps))

    if start is not None and start < common_ts[0]:
        raise ValueError('Start timestamp must be greater than common timestamp')
    if end is not None and end > common_ts[-1]:
        raise ValueError('End timestamp must be less than common timestamp')

    start = start if start is not None else common_ts[0]
    end = end if end is not None else common_ts[-1]

    common_ts = [ts for ts in common_ts if start <= ts <= end]

    historical_values = [common_ts, ]
    for data in historical_list:
        ts, vals = data[:, 0], data[:, 1]
        val_map = dict(zip(ts, vals))

        aligned_row = [val_map[t] for t in common_ts]
        historical_values.append(aligned_row)

    historical_matrix = numpy.array(historical_values).T
    forecast_matrix = None
    if predict is not None:
        check = None
        forecast_values = [common_ts, ]
        for data in forecast_list:
            ts, vals, chk = data[:, predict, 0], data[:, predict, 2], data[:, predict, 1]

            chk = chk.tolist()
            if check is None:
                check = chk
            else:
                if check != chk:
                    raise ValueError()

            val_map = dict(zip(ts, vals))
            aligned_row = [val_map[t] for t in common_ts]
            forecast_values.append(aligned_row)
        forecast_matrix = numpy.array(forecast_values).T

    return historical_matrix, forecast_matrix


def different_parameters(result_file: Path):
    with result_file.open('r') as f:
        result = json.load(f)

    limitation = 60

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}

        # Lowest
        if str((0,)) not in result['data']:
            futures[(0,)] = executor.submit(
                Strategy.evaluate,
                datacenters=result['meta']['regions']
            )

        # Static
        for setup in range(5, 5 + limitation, 5):
            for hold in range(5, 5 + limitation, 5):
                config = (setup * 60, hold * 60)
                if str(config) not in result['data']:
                    futures[config] = executor.submit(
                        Strategy.evaluate,
                        datacenters=result['meta']['regions'],
                        setup_time=setup * 60,
                        hold_time=hold * 60,
                    )

                for pred in range(1, 16):
                    config = (setup * 60, hold * 60, 0, pred * 5 * 60)
                    if str(config) not in result['data']:
                        futures[config] = executor.submit(
                            Strategy.evaluate,
                            datacenters=result['meta']['regions'],
                            setup_time=setup * 60,
                            hold_time=hold * 60,
                            predict=pred,
                        )

                # Dynamic
                for co in range(1, 16):
                    config = (setup * 60, hold * 60, co * 60)
                    if str(config) not in result['data']:
                        futures[config] = executor.submit(
                            Strategy.evaluate,
                            datacenters=result['meta']['regions'],
                            setup_time=setup * 60,
                            hold_time=hold * 60,
                            coexist_time=co * 60,
                        )

                    for pred in range(1, 16):
                        config = (setup * 60, hold * 60, co * 60, pred * 5 * 60)
                        if str(config) not in result['data']:
                            futures[config] = executor.submit(
                                Strategy.evaluate,
                                datacenters=result['meta']['regions'],
                                setup_time=setup * 60,
                                hold_time=hold * 60,
                                coexist_time=co * 60,
                                predict=pred,
                            )

        for config, future in futures.items():
            result['data'][str(config)] = future.result()
            print(config, future.result())

    with result_file.open('w') as f:
        json.dump(result, f, indent=2)
