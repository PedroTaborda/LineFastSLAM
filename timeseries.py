from typing import Any, Callable, Tuple, Union, Iterator

import numpy as np

class Timeseries:
    def __init__(self, data: np.ndarray, timestamps: np.ndarray):
        if data.shape[0] != timestamps.shape[0]:
            raise ValueError('Timeseries data and timestamps must have the same length')
        self.data: np.ndarray = data
        self.t: np.ndarray = timestamps
        self.shape: Tuple = data.shape

    def __getitem__(self, key: Any) -> Union[np.ndarray, 'Timeseries']:
        if isinstance(key, slice):
            return Timeseries(self.data[key], self.t[key])
        elif isinstance(key, int):
            return self.data[key]
        elif isinstance(key, tuple):
            return Timeseries(self.data[key], self.t[key])
        else:
            raise TypeError('Unsupported key type')

    def __len__(self) -> int:
        return self.data.shape[0]

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return zip(self.data, self.t)

    def __repr__(self) -> str:
        return f'Timeseries(data={self.data}, t={self.t})'

    def __str__(self) -> str:
        return f'Timeseries(data={self.data}, t={self.t})'

    def __call__(self, func: Callable[[np.ndarray], np.ndarray]) -> 'Timeseries':
        return Timeseries(func(self.data), self.t)

    @staticmethod
    def _apply(series1: 'Timeseries', series2: 'Timeseries', f: Callable[[Any, Any], Any]) -> 'Timeseries':
        """Applies a function f to two timeseries in the window where both
        timeseries coexist.
        """
        t = max(series1.t[0], series2.t[0])
        series_out_data = []
        series_out_t = []

        i1 = np.searchsorted(series1.t, t, side='left')
        i2 = np.searchsorted(series2.t, t, side='left')
        while t < min(series1.t[-1], series2.t[-1]):
            series_out_data.append(f(series1.data[i1], series2.data[i2]))
            series_out_t.append(t)
            print(f'{t} {series1.t[i1]} {series2.t[i2]}')
            print(f'i1={i1} i2={i2}')
            if i1 < series1.t.shape[0]-1 and i2 < series2.t.shape[0]-1:
                if series1.t[i1+1] < series2.t[i2+1]:
                    i1 += 1
                elif series1.t[i1+1] > series2.t[i2+1]:
                    i2 += 1
                else:
                    i1 += 1
                    i2 += 1
            elif i1 < series1.t.shape[0]-1:
                i1 += 1
            elif i2 < series2.t.shape[0]-1:
                i2 += 1
            else:
                break

        return Timeseries(np.array(series_out_data), np.array(series_out_t))


if __name__ == '__main__':
    x = Timeseries(np.array([1, 1, 0]), np.array([0, 1, 2]))
    y = Timeseries(np.array([1, 2, 2]), np.array([1, 2, 3]))

    z = Timeseries._apply(x, y, lambda x, y: x + y)