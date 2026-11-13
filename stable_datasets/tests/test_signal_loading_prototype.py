"""Correctness checks for the isolated physiological-signal prototype."""

import numpy as np
import pyarrow as pa
import pytest
import torch

from profiling.signal_loading import calibrate, calibration_parameters, extract_windows


SIGNAL_TYPE = pa.list_(pa.list_(pa.float32()))


def test_windows_preserve_order_duplicates_and_chunk_slice_offsets():
    recordings = [np.arange(n * 3, dtype=np.float32).reshape(n, 3) + i * 100 for i, n in enumerate([5, 7, 4, 6])]
    array = pa.array([x.tolist() for x in recordings], type=SIGNAL_TYPE)
    column = pa.chunked_array([array.slice(1, 1), array.slice(2, 2)])
    requests = [(2, 3), (0, 1), (1, 0), (2, 3)]
    expected = np.stack([recordings[row + 1][start : start + 2] for row, start in requests])
    actual = extract_windows(column, requests, length=2, channels=3)
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.writeable and actual.flags.c_contiguous
    actual[0, 0, 0] = -99
    assert column[2].as_py()[3][0] != -99


def test_windows_support_large_lists_and_empty_chunks():
    dtype = pa.large_list(pa.large_list(pa.int16()))
    chunk = pa.array([[[1, 2], [3, 4]], [[5, 6]]], type=dtype)
    column = pa.chunked_array([pa.array([], type=dtype), chunk.slice(1)])
    np.testing.assert_array_equal(extract_windows(column, [(0, 0)], 1, 2), [[[5, 6]]])
    assert extract_windows(column, [], 3, 2).shape == (0, 3, 2)


@pytest.mark.parametrize("requests", [[(-1, 0)], [(1, 0)], [(0, -1)], [(0, 1)]])
def test_windows_reject_out_of_bounds(requests):
    column = pa.array([[[1, 2], [3, 4]]], type=SIGNAL_TYPE)
    with pytest.raises(IndexError):
        extract_windows(column, requests, 2, 2)


@pytest.mark.parametrize(
    "recording,match",
    [(None, "recording is null"), ([None], "null time"), ([[1, None]], "null samples"), ([[1]], "channel count")],
)
def test_windows_reject_nulls_and_wrong_channels(recording, match):
    column = pa.array([recording], type=SIGNAL_TYPE)
    with pytest.raises(ValueError, match=match):
        extract_windows(column, [(0, 0)], 1, 2)


def test_nulls_outside_selected_window_do_not_poison_selection():
    column = pa.array([None, [None, [1, 2], [3, 4], [None, None]]], type=SIGNAL_TYPE)
    np.testing.assert_array_equal(extract_windows(column, [(1, 1)], 2, 2), [[[1, 2], [3, 4]]])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_calibration_matches_independent_physical_reference(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")
    dmin = np.array([-32768, -2048, 0])
    dmax = np.array([32767, 2047, 1000])
    pmin = np.array([-200, 50, 5])
    pmax = np.array([300, -50, 20])
    values = np.stack([dmin, dmax, [0, -50, 400]]).astype(np.int16)[None]
    gain, offset = calibration_parameters(dmin, dmax, pmin, pmax)
    actual = (
        calibrate(
            torch.from_numpy(values).to(device), torch.from_numpy(gain).to(device), torch.from_numpy(offset).to(device)
        )
        .cpu()
        .numpy()
    )
    expected = (values.astype(np.float64) - dmin) * (pmax - pmin) / (dmax - dmin) + pmin
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-4)


@pytest.mark.parametrize(
    "bounds",
    [
        ([], [], [], []),
        ([0], [0], [-1], [1]),
        ([0], [10], [1], [1]),
        ([0], [10, 20], [-1], [1]),
        ([0], [10], [np.nan], [1]),
    ],
)
def test_invalid_calibration_metadata_is_rejected(bounds):
    with pytest.raises(ValueError):
        calibration_parameters(*bounds)


def test_calibration_rejects_wrong_dtype_and_channel_count():
    with pytest.raises(ValueError, match="int16"):
        calibrate(torch.zeros(2, 3), torch.ones(3), torch.zeros(3))
    with pytest.raises(ValueError, match="channel count"):
        calibrate(torch.zeros(2, 3, dtype=torch.int16), torch.ones(2), torch.zeros(2))
