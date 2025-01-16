from typing import List
import json
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from numba import njit
from argparse import ArgumentParser

with open("/database/index/orbit2filepaths.json") as file:
    orbit2filepaths = json.load(file)


max_mss = [7.0e20, 2.45e20, 1.25e20]


@njit
def get_nen(radiance: np.ndarray, snr_coef: np.ndarray,
            max_ms: float) -> np.ndarray:
    c_photon = snr_coef[:, 0]
    c_background = snr_coef[:, 1]
    max_ms = max_ms / 1e20
    radiance = radiance / 1e20
    return max_ms / 100.0 * np.sqrt(
        np.abs(100*radiance/max_ms)*c_photon**2 + c_background**2
    ) * 1e20


@njit
def get_snr(radiance: np.ndarray, snr_coef: np.ndarray,
            max_ms: float) -> np.ndarray:
    c_photon = snr_coef[:, 0]
    c_background = snr_coef[:, 1]
    max_ms = max_ms / 1e20
    radiance = radiance / 1e20
    return np.sqrt(
        100 * radiance**2 / (max_ms * (c_background**2 *
                             max_ms/100 + c_photon**2 * radiance))
    )


@njit
def get_wl(dispersion_coef: np.ndarray):
    columns = np.arange(1, 1017, 1, dtype=np.float64)
    ans = np.zeros_like(columns)
    for i, coef in enumerate(dispersion_coef):
        ans += columns**i * coef
    return ans

# def get_spectrum(sounding_indexs: np.ndarray, snr_coef: np.ndarray,
#                  bsl: np.ndarray, dcs: np.ndarray, full_radiances: np.ndarray,
#                  max_ms: float):
#     wls = np.array([
#         get_wl(dcs[sidx]) for sidx in sounding_indexs
#     ])
#     rads = np.array([
#         full_radiances[fidx, sidx, :] for fidx, sidx in enumerate(sounding_indexs)
#     ])
#     snrs = np.array(
#         [get_snr(rads[i, :], snr_coef[sidx, :, :], max_ms) for i, sidx in enumerate(sounding_indexs)]
#     )
#     bsls = np.array(
#         [bsl[sidx, :] for sidx in sounding_indexs]
#     )
#     return np.hstack([wls, rads, snrs, bsls])


@njit
def get_spectrum(sounding_indexs: np.ndarray, snr_coef: np.ndarray,
                 bsl: np.ndarray, dcs: np.ndarray, full_radiances: np.ndarray,
                 max_ms: float) -> np.ndarray:
    num_sounding = len(sounding_indexs)
    num_columns = 1016
    wls = np.zeros((num_sounding, num_columns))
    rads = np.zeros((num_sounding, num_columns))
    snrs = np.zeros((num_sounding, num_columns))
    bsls = np.zeros((num_sounding, num_columns))

    for i in range(num_sounding):
        sidx = sounding_indexs[i]
        wls[i, :] = get_wl(dcs[sidx])
        rads[i, :] = full_radiances[i, sidx, :]
        snrs[i, :] = get_snr(rads[i, :], snr_coef[sidx, :, :], max_ms)
        bsls[i, :] = bsl[sidx, :]
    return np.hstack((wls, rads, snrs, bsls))


@njit
def get_short_name(name: str):
    return name.split("/")[-1].replace("retrieval_", "")


def create_columns():
    features = [
        "/RetrievalGeometry/retrieval_latitude",
        "/RetrievalGeometry/retrieval_longitude",
        "/RetrievalGeometry/retrieval_solar_azimuth",
        "/RetrievalGeometry/retrieval_solar_zenith",
        "/RetrievalGeometry/retrieval_azimuth",
        "/RetrievalGeometry/retrieval_zenith",
        "/RetrievalGeometry/retrieval_solar_distance",
        "/RetrievalGeometry/retrieval_polarization_angle",
        "/RetrievalGeometry/retrieval_aspect",
        "/RetrievalGeometry/retrieval_slope",
        "/RetrievalGeometry/retrieval_solar_relative_velocity",
        "/RetrievalGeometry/retrieval_relative_velocity",
        "/RetrievalGeometry/retrieval_altitude",
        "/RetrievalResults/xco2_apriori",
        "/RetrievalResults/surface_pressure_apriori_fph",
    ]

    labels = [
        "/RetrievalResults/xco2",
        "/RetrievalResults/surface_pressure_fph",
        "/RetrievalResults/xco2_uncert",
    ]

    def get_short_name(name):
        return name.split("/")[-1].replace("retrieval_", "")

    band_names = ["o2", "weak_co2", "strong_co2"]
    array_names = ["wavelengths", "radiances", "snrs", "bad_sample_list"]

    columns = []

    for header, names in zip(["features", "labels"], [features, labels]):
        for name in names:
            name = get_short_name(name)
            columns.append((header, name))

    for array_name in array_names:
        for band_name in band_names:
            for pixel_index in range(1, 1017):
                columns.append(
                    (f"{array_name}_{band_name}", f"{pixel_index:04d}"))

    columns = pd.MultiIndex.from_tuples(columns)
    return band_names, features, labels, array_names, columns


def create_df(orbit: str, band_names: List[str], features: List[str],
              labels: List[str], array_names: List[str],
              columns: pd.MultiIndex, path: str, channel_index:int = 1):
    l1b_name, l2d_name, met_name = orbit2filepaths[orbit]
    with h5py.File(l2d_name) as l2d, h5py.File(l1b_name) as l1b:
        sounding_ids = l2d["/RetrievalHeader/sounding_id"][:]
        # Indicator of retrieval results:
        # 1 - 'Passed internal quality check',
        # 2 - 'Failed internal qualiy check',
        # 3 - 'Reached maximum alowed iterations',
        # 4 - 'Reached maximum allowed divergences'.
        outcome_flag = l2d["/RetrievalResults/outcome_flag"][:]
        retrieval_land_fraction = l2d["/RetrievalGeometry/retrieval_land_fraction"][:]
        between = lambda x, low, high: (x > low) & (x < high)
        latitudes = l2d["/RetrievalGeometry/retrieval_latitude"][:]
        longitudes = l2d["/RetrievalGeometry/retrieval_longitude"][:]
        l2d_indices = (outcome_flag == 1) & (
            retrieval_land_fraction == 100) & (
            sounding_ids % 10 == channel_index) & (
            between(latitudes, 20, 45)) & (
            between(longitudes, 110, 145))

        if np.sum(l2d_indices) == 0:
            return 0

        sounding_ids = l2d["/RetrievalHeader/sounding_id"][l2d_indices]
        frame_indexs = l2d["/RetrievalHeader/frame_index"][l2d_indices] - 1
        sounding_indexs = l2d["/RetrievalHeader/sounding_index"][l2d_indices] - 1

        data = np.full([len(sounding_ids), len(columns)], np.nan)
        df = pd.DataFrame(data, index=sounding_ids, columns=columns)

        for header, names in zip(["features", "labels"], [features, labels]):
            for name in names:
                data = l2d[name][l2d_indices]
                column_name = get_short_name(name)
                df[(header, column_name)] = data

        snr_coefs = l1b["InstrumentHeader/snr_coef"][:]
        bad_sample_list = l1b["InstrumentHeader/bad_sample_list"][:]
        dispersion_coef_samp = l1b["InstrumentHeader/dispersion_coef_samp"][:]
        # cidx - channel index, fidx - frame index, sidx - sounding index (not sounding_id)
        for cidx, band_name in enumerate(band_names):
            snr_coef = snr_coefs[cidx, :]
            bsl = bad_sample_list[cidx, :]
            dcs = dispersion_coef_samp[cidx, :]
            full_radiances = l1b[f"SoundingMeasurements/radiance_{band_name}"][frame_indexs, :]
            ans = get_spectrum(
                sounding_indexs, snr_coef, bsl, dcs, full_radiances, max_mss[cidx]
            )

            cur_columns = [
                f"{array_name}_{band_name}" for array_name in array_names]
            df.loc[:, cur_columns] = ans
    df.to_parquet(f"{path}/{channel_index}_{orbit}.parquet")


def main():
    parser = ArgumentParser()
    parser.add_argument("worker", type=str)
    parser.add_argument("channel_index", type=str)
    args = parser.parse_args()
    worker = args.worker
    channel_index = int(args.channel_index)
    
    with open("orbits.json") as file:
        worker2orbits = json.load(file)
    orbits = worker2orbits[worker]
    
    path = "/fast_read/new_data"
    band_names, features, labels, array_names, columns = create_columns()

    for orbit in tqdm(orbits):
        create_df(orbit, band_names, features,
                  labels, array_names, columns, path, channel_index)


if __name__ == "__main__":
    main()
