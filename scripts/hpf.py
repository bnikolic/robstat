import multiprocess as multiprocessing
import os

import numpy as np

import uvtools

from robstat.utils import DATAPATH


def main():
    # xd_vis_file = 'xd_vis_rph.npz'
    xd_vis_file = 'lstb_no_avg/idr2_lstb_14m_ee_1.40949.npz'

    xd_vis_file_path = os.path.join(DATAPATH, xd_vis_file)
    hpf_vis_file = os.path.join(DATAPATH, xd_vis_file.replace('.npz', '_hpf.npz'))

    if not os.path.exists(hpf_vis_file):

        # load dataset
        sample_xd_data = np.load(xd_vis_file_path)

        xd_data = sample_xd_data['data'] # dimensions (days, freqs, times, bls)
        xd_flags = sample_xd_data['flags']

        xd_redg = sample_xd_data['redg']
        xd_times = sample_xd_data['times']
        xd_pol = sample_xd_data['pol'].item()
        JDs = sample_xd_data['JDs']

        freqs = sample_xd_data['freqs']
        chans = sample_xd_data['chans']

        f_resolution = np.median(np.ediff1d(freqs))
        no_chans = chans.size
        no_days = JDs.size
        no_tints = xd_times.size
        no_bls = xd_data.shape[3]


        # HPF with DAYENU
        filter_centers = [0.] # center of rectangular fourier regions to filter
        filter_half_widths = [1e-6] # half-width of rectangular fourier regions to filter
        mode = 'dayenu_dpss_leastsq'

        def bl_iter(bl):
            hpf_data_d = np.empty((no_days, no_chans, no_tints), dtype=complex)
            for day in range(no_days):
                data = xd_data[day, ..., bl]
                flgs = xd_flags[day, ..., bl]

                if flgs.all():
                    d_res_d = np.empty_like(data) * np.nan
                else:
                    wgts = np.logical_not(flgs).astype(float)

                    _, d_res_d, _ = uvtools.dspec.fourier_filter(freqs, data, wgts, filter_centers, \
                        filter_half_widths, mode, filter_dims=0, skip_wgt=0., zero_residual_flags=True, \
                        max_contiguous_edge_flags=500)

                hpf_data_d[day, ...] = d_res_d

            return  hpf_data_d[..., np.newaxis]

        m_pool = multiprocessing.Pool(min(multiprocessing.cpu_count(), no_bls))
        hpf_data = np.concatenate(m_pool.map(bl_iter, range(no_bls)), axis=3)


        # save results
        hpf_data[xd_flags] *= np.nan

        keys = list(sample_xd_data.keys())
        keys.remove('data')
        keys.remove('antpos')
        metadata = {k: sample_xd_data[k] for k in keys}
        metadata['antpos'] = np.load(xd_vis_file_path, allow_pickle=True)['antpos'].item()

        np.savez(hpf_vis_file, data=hpf_data, **metadata)
        print('HPF visibility file saved to: {}'.format(hpf_vis_file))


if __name__ == '__main__':
    main()
