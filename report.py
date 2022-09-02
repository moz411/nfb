import io
import os
import base64
import json
import numpy as np
import pandas as pd
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from google.cloud import storage
from matplotlib import cm, colors 
import matplotlib.patches as mpatches
from mne.time_frequency import psd_array_multitaper
from brainflow.board_shim import BoardShim

# Generate report
def report(uuid, sessionid, board_id):
    bucket = os.environ.get("BUCKET_NAME")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket)
    descr = BoardShim.get_board_descr(board_id)
    sampling_rate = descr['sampling_rate']
    # extract eeg_channels from board description
    # TODO: check compat with brainflow 5.1
    names = pd.Series(['package_num'] + descr['eeg_names'].split(',') + [pd.NA]*500)
    for key, val in descr.items():
        if key.endswith('_channel') and val > len(descr['eeg_channels']):
            names[val] = key.split('_')[0]
        if key.endswith('_channels') and val[0] > len(descr['eeg_channels']):
            for i in val:
                names[i] = '%s_%d' % (key.split('_')[0],i)
    names = names[~names.isna()].values
    eeg_names = descr['eeg_names'].split(',')

    # get all the data files from gcs and concatenate into a DataFrame
    data = pd.DataFrame(columns=names)
    file = '/tmp/%s.zip' % uuid
    blobs = storage_client.list_blobs(bucket, prefix='%s/%s/0' % (uuid, sessionid))
    for blob in blobs:
        blob.download_to_filename(file)
        df = pd.read_csv(file, sep='\t', names=names)
        data = pd.concat([data, df])

    # basic outliers filtering
    q_low = data.quantile(0.05)
    q_hi  = data.quantile(0.95)
    data = data[(data < q_hi) & (data > q_low)]
    data['timestamp'] = pd.to_datetime(data.timestamp, unit='ms')

    # resample and cleanup duplicate entries
    data = data.groupby('timestamp').mean().fillna(method="ffill")

    # filter on eeg channels 
    data = data[eeg_names]

    # aggregate all the channels
    meandata = data.apply(lambda x: x.mean(), axis=1).fillna(0)

    # Cut the timeserie if a big shift in voltage is visible 
    # at the beginning or the end of the recording
    # Might happend when manipulating the headset
    bigshifts = np.argpartition(meandata.pct_change(), 2)[:2]
    for shift in bigshifts:
        if shift < len(meandata) * .05:
            meandata = meandata[shift:]
        if shift > len(meandata) * .95:
            meandata = meandata[:shift]

    # bandpower
    bands = {'Delta': {'low': 0,  'high': 4},
            'Theta':  {'low': 4,  'high': 8},
            'Alpha':  {'low': 8,  'high': 13},
            'Beta':   {'low': 13, 'high': 32},
            'Gamma':  {'low': 32, 'high': 200}}

    psd, freqs = psd_array_multitaper(meandata, sfreq=sampling_rate, verbose=0)
    # to DataFrame in Decibels
    psd = pd.DataFrame(10 * np.log10(psd), index=freqs)
    psd.index.name = "Fréquences (Hz)"
    # rolling average
    psd = psd.rolling(sampling_rate).mean()

    # plotting to file
    ax = psd.plot(figsize=(15,5))
    ax.set_ylabel('Décibels (dB)')
    cmap = cm.get_cmap('Pastel1')
    patches = []

    for i, band in enumerate(bands.keys()):
        color = colors.rgb2hex(cmap(i))
        idx_delta = np.logical_and(psd.index >= bands[band]['low'], psd.index <= bands[band]['high'])
        # compute integral
        freq_res = psd.index[1] - psd.index[0]
        score = simpson(psd[idx_delta].fillna(0).values[:,0], dx=freq_res)
        bands[band] = score
        # plot
        plt.fill_between(x=psd.index, y1=psd.values[:,0], y2=0, where=idx_delta, color=color)
        patches.append(mpatches.Patch(color=color, label='%s: %d' % (band, score)))
        
    ax.legend(handles=patches)
    image = io.BytesIO()
    plt.savefig(image, bbox_inches='tight', format='png')
    image.seek(0)
    bands['bandpower'] = base64.b64encode(image.read()).decode()
    destination_blob_name = '%s/%s/results.json' % (uuid, sessionid)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(json.dumps(bands))
