import io
import os
import base64
import datetime
import json
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from google.cloud import storage
import google.auth
from matplotlib import cm, colors
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from mne.time_frequency import psd_array_multitaper
from brainflow.board_shim import BoardShim
from jinja2 import Environment, FileSystemLoader, select_autoescape

jinja_env = Environment(loader=FileSystemLoader('.'))

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

    # phase spectrum
    plt.plot(figsize=(15,5))
    for i, channel in enumerate(data.columns):
        plt.phase_spectrum(data[channel].fillna(0))
    phase_image = io.BytesIO()
    plt.savefig(phase_image, bbox_inches='tight', format='png')
    phase_image.seek(0)
    phase_image = base64.b64encode(phase_image.read()).decode()
    plt.clf()

    # boxplots
    sns.set(rc={"figure.figsize":(15, 10)})
    ax = sns.boxplot(whis=100, data=data, palette="Set3", order=data.columns.to_list().sort())
    ax.set_xlabel('Canaux EEG')
    ax.set_ylabel('Tension (??V)')
    boxplots = io.BytesIO()
    plt.savefig(boxplots, bbox_inches='tight', format='png')
    boxplots.seek(0)
    boxplots = base64.b64encode(boxplots.read()).decode()

    # Timeserie
    ax = data.plot(subplots=True, figsize=(15,15))
    ax[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    timeserie = io.BytesIO()
    plt.savefig(timeserie, bbox_inches='tight', format='png')
    timeserie.seek(0)
    timeserie = base64.b64encode(timeserie.read()).decode()



    # bandpower
    bands = {'Delta': {'low': 0,  'high': 4},
            'Theta':  {'low': 4,  'high': 8},
            'Alpha':  {'low': 8,  'high': 13},
            'Beta':   {'low': 13, 'high': 32},
            'Gamma':  {'low': 32, 'high': 200}}

    psd, freqs = psd_array_multitaper(meandata, sfreq=sampling_rate, verbose=0)
    # to DataFrame in Decibels
    psd = pd.DataFrame(10 * np.log10(psd), index=freqs)
    psd.index.name = "Fr??quences (Hz)"
    # rolling average
    psd = psd.rolling(sampling_rate).mean()

    # plotting to file
    ax = psd.plot(figsize=(15,5))
    ax.set_ylabel('D??cibels (dB)')
    cmap = cm.get_cmap('Pastel1')
    patches = []

    for i, band in enumerate(bands.keys()):
        color = colors.rgb2hex(cmap(i))
        idx_delta = np.logical_and(psd.index >= bands[band]['low'], psd.index <= bands[band]['high'])
        # compute integral
        freq_res = psd.index[1] - psd.index[0]
        score = simpson(psd[idx_delta].fillna(0).values[:,0], dx=freq_res)
        bands[band] = round(score, 2)
        # plot
        plt.fill_between(x=psd.index, y1=psd.values[:,0], y2=0, where=idx_delta, color=color)
        patches.append(mpatches.Patch(color=color, label='%s: %d' % (band, score)))
        
    ax.legend(handles=patches)
    bandpower = io.BytesIO()
    plt.savefig(bandpower, bbox_inches='tight', format='png')
    bandpower.seek(0)
    bandpower = base64.b64encode(bandpower.read()).decode()

    # generate report
    template = jinja_env.get_template("report.html")
    res = template.render(title="nfb@moz42.net", bands=bands, bandpower=bandpower, phase_spectrum=phase_image,
                          timeserie=timeserie, boxplots=boxplots)

    # save report to GCS
    destination_blob_name = '%s/%s/report.html' % (uuid, sessionid)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(res)

    # save results to GCS
    # destination_blob_name = '%s/%s/results.json' % (uuid, sessionid)
    # blob = bucket.blob(destination_blob_name)
    # blob.upload_from_string(json.dumps(bands))
    
    bands['phase_spectrum'] = phase_image # to be returned in payload from POST request
    return(json.dumps(bands))
