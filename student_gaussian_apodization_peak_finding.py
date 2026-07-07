

"""
Minimal teaching helpers for Gaussian apodization and 2D correlation peak finding.

Only two public functions are intended for students to call:

  - gaussian_apodization_window(...)
  - find_correlation_peak_2d(...)

The small private lookup helper below is included only because the Gaussian
window uses the same MATLAB-style ``findwidth`` table as the analysis code.
"""

from __future__ import annotations

import numpy as np


_FINDWIDTH_R = np.array(
    [
        0,
        0.0051,
        0.0052,
        0.0053,
        0.0055,
        0.0056,
        0.0057,
        0.0059,
        0.006,
        0.0063,
        0.0064,
        0.0066,
        0.0067,
        0.0069,
        0.007,
        0.0072,
        0.0074,
        0.0076,
        0.0079,
        0.0081,
        0.0083,
        0.0085,
        0.0087,
        0.0089,
        0.0091,
        0.0093,
        0.0095,
        0.01,
        0.0102,
        0.0104,
        0.0107,
        0.0109,
        0.0112,
        0.0114,
        0.0117,
        0.012,
        0.0125,
        0.0128,
        0.0131,
        0.0134,
        0.0137,
        0.0141,
        0.0144,
        0.0147,
        0.0151,
        0.0158,
        0.0161,
        0.0165,
        0.0169,
        0.0173,
        0.0177,
        0.0181,
        0.0185,
        0.019,
        0.0199,
        0.0203,
        0.0208,
        0.0213,
        0.0218,
        0.0223,
        0.0228,
        0.0233,
        0.0239,
        0.025,
        0.0256,
        0.0262,
        0.0268,
        0.0274,
        0.0281,
        0.0287,
        0.0294,
        0.0301,
        0.0315,
        0.0322,
        0.033,
        0.0337,
        0.0345,
        0.0353,
        0.0361,
        0.037,
        0.0378,
        0.0396,
        0.0406,
        0.0415,
        0.0425,
        0.0435,
        0.0445,
        0.0455,
        0.0466,
        0.0476,
        0.0499,
        0.0511,
        0.0522,
        0.0535,
        0.0547,
        0.056,
        0.0573,
        0.0586,
        0.06,
        0.0628,
        0.0643,
        0.0658,
        0.0673,
        0.0689,
        0.0705,
        0.0721,
        0.0738,
        0.0755,
        0.0791,
        0.0809,
        0.0828,
        0.0847,
        0.0867,
        0.0887,
        0.0908,
        0.0929,
        0.0951,
        0.0996,
        0.1019,
        0.1042,
        0.1067,
        0.1092,
        0.1117,
        0.1143,
        0.117,
        0.1197,
        0.1253,
        0.1283,
        0.1312,
        0.1343,
        0.1374,
        0.1406,
        0.1439,
        0.1473,
        0.1507,
        0.1578,
        0.1615,
        0.1652,
        0.1691,
        0.173,
        0.177,
        0.1812,
        0.1854,
        0.1897,
        0.1986,
        0.2033,
        0.208,
        0.2128,
        0.2178,
        0.2229,
        0.2281,
        0.2334,
        0.2388,
        0.2501,
        0.2559,
        0.2619,
        0.268,
        0.2742,
        0.2806,
        0.2871,
        0.2938,
        0.3006,
        0.3148,
        0.3221,
        0.3296,
        0.3373,
        0.3451,
        0.3531,
        0.3613,
        0.3696,
        0.3781,
        0.3957,
        0.4048,
        0.414,
        0.4233,
        0.4329,
        0.4425,
        0.4524,
        0.4623,
        0.4724,
        0.493,
        0.5034,
        0.5139,
        0.5244,
        0.5351,
        0.5457,
        0.5564,
        0.5672,
        0.5779,
        0.5992,
        0.6099,
        0.6204,
        0.6309,
        0.6414,
        0.6517,
        0.6619,
        0.672,
        0.6819,
        0.7014,
        0.7109,
        0.7203,
        0.7295,
        0.7385,
        0.7473,
        0.7559,
        0.7643,
        0.7726,
        0.7884,
        0.796,
        0.8035,
        0.8107,
        0.8177,
        0.8245,
        0.8311,
        0.8376,
        0.8438,
        0.8556,
        0.8613,
        0.8667,
        0.872,
        0.8771,
        0.882,
        0.8867,
        0.8913,
        0.8957,
        0.9041,
        0.908,
        0.9118,
        0.9155,
        0.919,
        0.9224,
        0.9256,
        0.9288,
        0.9318,
        0.9374,
        0.9401,
        0.9426,
        0.9451,
        0.9474,
        0.9497,
        0.9519,
        0.9539,
        0.9559,
        0.9597,
        0.9614,
        0.9631,
        0.9647,
        0.9662,
        0.9677,
        0.9691,
        0.9705,
        0.9718,
        0.9742,
        0.9753,
        0.9764,
        0.9775,
        0.9785,
        0.9794,
        0.9803,
        0.9812,
        0.982,
        0.9836,
        0.9843,
        0.985,
        0.9857,
        0.9863,
        0.9869,
        0.9875,
        0.9881,
        0.9886,
        0.9896,
        0.99,
        0.9905,
        0.9909,
        0.9913,
        0.9917,
        0.9921,
        0.9924,
        0.9928,
        0.9934,
        0.9937,
        0.994,
        0.9943,
        0.9945,
        0.9948,
        0.995,
        1,
        np.inf,
    ],
    dtype=float,
)

_FINDWIDTH_P = np.array(
    [
        500,
        245.4709,
        239.8833,
        234.4229,
        229.0868,
        223.8721,
        218.7762,
        213.7962,
        208.9296,
        199.5262,
        194.9845,
        190.5461,
        186.2087,
        181.9701,
        177.8279,
        173.7801,
        169.8244,
        165.9587,
        158.4893,
        154.8817,
        151.3561,
        147.9108,
        144.544,
        141.2538,
        138.0384,
        134.8963,
        131.8257,
        125.8925,
        123.0269,
        120.2264,
        117.4898,
        114.8154,
        112.2018,
        109.6478,
        107.1519,
        104.7129,
        100,
        97.7237,
        95.4993,
        93.3254,
        91.2011,
        89.1251,
        87.0964,
        85.1138,
        83.1764,
        79.4328,
        77.6247,
        75.8578,
        74.131,
        72.4436,
        70.7946,
        69.1831,
        67.6083,
        66.0693,
        63.0957,
        61.6595,
        60.256,
        58.8844,
        57.544,
        56.2341,
        54.9541,
        53.7032,
        52.4807,
        50.1187,
        48.9779,
        47.863,
        46.7735,
        45.7088,
        44.6684,
        43.6516,
        42.658,
        41.6869,
        39.8107,
        38.9045,
        38.0189,
        37.1535,
        36.3078,
        35.4813,
        34.6737,
        33.8844,
        33.1131,
        31.6228,
        30.903,
        30.1995,
        29.5121,
        28.8403,
        28.1838,
        27.5423,
        26.9153,
        26.3027,
        25.1189,
        24.5471,
        23.9883,
        23.4423,
        22.9087,
        22.3872,
        21.8776,
        21.3796,
        20.893,
        19.9526,
        19.4984,
        19.0546,
        18.6209,
        18.197,
        17.7828,
        17.378,
        16.9824,
        16.5959,
        15.8489,
        15.4882,
        15.1356,
        14.7911,
        14.4544,
        14.1254,
        13.8038,
        13.4896,
        13.1826,
        12.5893,
        12.3027,
        12.0226,
        11.749,
        11.4815,
        11.2202,
        10.9648,
        10.7152,
        10.4713,
        10,
        9.7724,
        9.5499,
        9.3325,
        9.1201,
        8.9125,
        8.7096,
        8.5114,
        8.3176,
        7.9433,
        7.7625,
        7.5858,
        7.4131,
        7.2444,
        7.0795,
        6.9183,
        6.7608,
        6.6069,
        6.3096,
        6.166,
        6.0256,
        5.8884,
        5.7544,
        5.6234,
        5.4954,
        5.3703,
        5.2481,
        5.0119,
        4.8978,
        4.7863,
        4.6774,
        4.5709,
        4.4668,
        4.3652,
        4.2658,
        4.1687,
        3.9811,
        3.8905,
        3.8019,
        3.7154,
        3.6308,
        3.5481,
        3.4674,
        3.3884,
        3.3113,
        3.1623,
        3.0903,
        3.02,
        2.9512,
        2.884,
        2.8184,
        2.7542,
        2.6915,
        2.6303,
        2.5119,
        2.4547,
        2.3988,
        2.3442,
        2.2909,
        2.2387,
        2.1878,
        2.138,
        2.0893,
        1.9953,
        1.9498,
        1.9055,
        1.8621,
        1.8197,
        1.7783,
        1.7378,
        1.6982,
        1.6596,
        1.5849,
        1.5488,
        1.5136,
        1.4791,
        1.4454,
        1.4125,
        1.3804,
        1.349,
        1.3183,
        1.2589,
        1.2303,
        1.2023,
        1.1749,
        1.1482,
        1.122,
        1.0965,
        1.0715,
        1.0471,
        1,
        0.9772,
        0.955,
        0.9333,
        0.912,
        0.8913,
        0.871,
        0.8511,
        0.8318,
        0.7943,
        0.7762,
        0.7586,
        0.7413,
        0.7244,
        0.7079,
        0.6918,
        0.6761,
        0.6607,
        0.631,
        0.6166,
        0.6026,
        0.5888,
        0.5754,
        0.5623,
        0.5495,
        0.537,
        0.5248,
        0.5012,
        0.4898,
        0.4786,
        0.4677,
        0.4571,
        0.4467,
        0.4365,
        0.4266,
        0.4169,
        0.3981,
        0.389,
        0.3802,
        0.3715,
        0.3631,
        0.3548,
        0.3467,
        0.3388,
        0.3311,
        0.3162,
        0.309,
        0.302,
        0.2951,
        0.2884,
        0.2818,
        0.2754,
        0.2692,
        0.263,
        0.2512,
        0.2455,
        0.2399,
        0.2344,
        0.2291,
        0.2239,
        0.2188,
        0.2138,
        0.2089,
        0.1995,
        0.195,
        0.1905,
        0.1862,
        0.182,
        0.1778,
        0.1738,
        0,
        0,
    ],
    dtype=float,
)


def _findwidth_exact(window_ratio: float) -> float:
    """Private MATLAB-compatible lookup used by gaussian_apodization_window."""

    ratio = float(window_ratio)
    if ratio <= _FINDWIDTH_R[0]:
        return float(_FINDWIDTH_P[0])

    finite = np.isfinite(_FINDWIDTH_R)
    max_ratio = float(np.max(_FINDWIDTH_R[finite]))
    if ratio >= max_ratio:
        return 0.0

    return float(np.interp(ratio, _FINDWIDTH_R[finite], _FINDWIDTH_P[finite]))


def gaussian_apodization_window(
    dimensions: tuple[int, int],
    window_size: tuple[float, float] = (0.5, 0.5),
    window_type: str = "fraction",
) -> np.ndarray:
    """Build a 2D Gaussian apodization window.

    Args:
        dimensions:
            Output shape as ``(height, width)``.
        window_size:
            If ``window_type='fraction'``, this is
            ``(height_fraction, width_fraction)``. If ``window_type='pixels'``,
            this is ``(height_pixels, width_pixels)``.
        window_type:
            Either ``'fraction'`` or ``'pixels'``.

    Returns:
        A ``height x width`` Gaussian weighting window.
    """

    height, width = int(dimensions[0]), int(dimensions[1])
    if height <= 0 or width <= 0:
        raise ValueError("dimensions must contain positive height and width")

    win_y, win_x = float(window_size[0]), float(window_size[1])
    if window_type == "fraction":
        window_size_y = height * win_y
        window_size_x = width * win_x
    elif window_type == "pixels":
        window_size_y = win_y
        window_size_x = win_x
    else:
        raise ValueError("window_type must be either 'fraction' or 'pixels'")

    py = _findwidth_exact(window_size_y / float(height))
    px = _findwidth_exact(window_size_x / float(width))

    x = np.linspace(-1.0, 1.0, width, dtype=float)
    y = np.linspace(-1.0, 1.0, height, dtype=float)[:, None]
    wx = np.exp(-(px**2) * (x**2) / 2.0)
    wy = np.exp(-(py**2) * (y**2) / 2.0)
    return np.asarray(wy @ wx[None, :], dtype=float)


def find_correlation_peak_2d(correlation: np.ndarray, fit_weight: np.ndarray | None = None) -> dict[str, float | int]:
    """Find a subpixel peak in a 2D correlation surface.

    This readable version finds the integer maximum, then applies a three-point
    log-parabolic fit in x and y. It is suitable for teaching the idea used in
    point tracking. Production code may use a heavier MATLAB-parity nonlinear
    Gaussian fit around the same peak.

    Args:
        correlation:
            2D correlation surface, typically after FFT-based correlation and
            ``fftshift``.
        fit_weight:
            Optional 2D weighting window used only for the local subpixel fit.

    Returns:
        A dictionary with:
          - ``u``: horizontal displacement in pixels
          - ``v``: vertical displacement in pixels
          - ``peak_value``: integer peak value
          - ``row_0based``: row index of the integer peak
          - ``col_0based``: column index of the integer peak
    """

    def log_parabolic_offset(left: float, center: float, right: float) -> float:
        if left <= 0.0 or center <= 0.0 or right <= 0.0:
            return 0.0
        log_left = float(np.log(left))
        log_center = float(np.log(center))
        log_right = float(np.log(right))
        denominator = 2.0 * (log_left + log_right - 2.0 * log_center)
        if abs(denominator) < 1e-12:
            return 0.0
        return float(np.clip((log_left - log_right) / denominator, -1.0, 1.0))

    corr = np.asarray(correlation, dtype=float)
    if corr.ndim != 2 or corr.size == 0:
        raise ValueError("correlation must be a non-empty 2D array")
    if not np.any(np.isfinite(corr)):
        raise ValueError("correlation must contain at least one finite value")

    if fit_weight is None:
        weight = np.ones_like(corr, dtype=float)
    else:
        weight = np.asarray(fit_weight, dtype=float)
        if weight.shape != corr.shape:
            raise ValueError("fit_weight must match correlation shape")

    corr_for_peak = corr.copy()
    corr_for_peak[~np.isfinite(corr_for_peak)] = -np.inf
    row0, col0 = np.unravel_index(int(np.argmax(corr_for_peak)), corr.shape)

    local = corr * weight
    dx = 0.0
    dy = 0.0
    if 0 < col0 < corr.shape[1] - 1:
        dx = log_parabolic_offset(local[row0, col0 - 1], local[row0, col0], local[row0, col0 + 1])
    if 0 < row0 < corr.shape[0] - 1:
        dy = log_parabolic_offset(local[row0 - 1, col0], local[row0, col0], local[row0 + 1, col0])

    x_axis = np.arange(-np.floor(corr.shape[1] / 2.0), np.ceil(corr.shape[1] / 2.0), dtype=float)
    y_axis = np.arange(-np.floor(corr.shape[0] / 2.0), np.ceil(corr.shape[0] / 2.0), dtype=float)

    return {
        "u": float(x_axis[col0] + dx),
        "v": float(y_axis[row0] + dy),
        "peak_value": float(corr[row0, col0]),
        "row_0based": int(row0),
        "col_0based": int(col0),
    }
