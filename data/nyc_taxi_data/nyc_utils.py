import pandas as pd
import numpy as np


def detect_topk_events(anomaly_scores, ratio):
    """Detect top-k events from the anomaly score tensor

    Args:
        anomaly_scores (np.array): Anomaly score tensor of dimension (81, 53, 7, 24)
        ratio (float): Ratio of events to detect.
    """
    events_start_ts = pd.to_datetime(['01-Jan-2018', '03-Jan-2018 16:00:00', '14-Jan-2018 09:00:00', '20-Jan-2018 08:00:00', 
                                    '4-Mar-2018 15:00:00', '08-Mar-2018 18:00:00', '17-Mar-2018 11:00:00', '20-Mar-2018 10:00:00',
                                    '21-Mar-2018 16:00:00', '01-Jul-2018 17:00:00', '04-Jul-2018 17:00:00', '25-Sep-2018 10:00:00',
                                    '04-Oct-2018 08:00:00', '04-Nov-2018 12:00:00', '09-Nov-2018 19:00:00', '22-Nov-2018 21:00:00',
                                    '4-Dec-2018 19:00:00', '16-Dec-2018 10:00:00', '28-Dec-2018 12:00:00', '31-Dec-2018 21:00:00',
                                    ], format='mixed')
    events_end_ts = pd.to_datetime(['01-Jan-2018 02:00:00', '03-Jan-2018 22:00:00', '14-Jan-2018 17:00:00', '20-Jan-2018 15:00:00',
                             '4-Mar-2018 22:00:00', '08-Mar-2018 23:59:00', '17-Mar-2018 17:00:00', '20-Mar-2018 20:00:00',
                            '21-Mar-2018 22:00:00', '01-Jul-2018 22:00:00', '04-Jul-2018 23:00:00', '25-Sep-2018 20:00:00',
                            '04-Oct-2018 15:00:00', '04-Nov-2018 17:00:00', '09-Nov-2018 23:30:00', '22-Nov-2018 23:59:00',
                            '4-Dec-2018 23:59:00', '16-Dec-2018 15:00:00', '28-Dec-2018 17:00:00', '31-Dec-2018 23:59:00'
                                ], format='mixed')
    indd = np.flip(np.argsort(anomaly_scores, axis=None))
    ind = np.unravel_index(indd[:int(len(indd)*ratio)], anomaly_scores.shape)
    topk_event_idx = ind
    anomaly_mask = np.zeros(anomaly_scores.shape, dtype=bool)
    anomaly_mask[topk_event_idx] =1
#     num_detected_events = 0
    detected_events = np.zeros(20)

    idxs = np.arange(81)
    # w = events_start_ts.isocalendar().week
    # d = events_start_ts.day_of_week
    doy = events_start_ts.day_of_year
    w = (doy-1)//(7)
    d = (doy-1) % 7
    h_s = events_start_ts.hour
    h_e = events_end_ts.hour
    for i in range(20):
        event_mask = np.zeros(anomaly_scores.shape, dtype=bool)
        locations = dates['dates'][2].ravel()[i].ravel()
        
        for loc in locations: 
            # event_mask[idxs[regions==loc], w[i]-1, d[i], h_s[i]:h_e[i]] = 1
            # event_mask[idxs[regions==loc], w[i]-1, d[i], h_e[i]] = 1
            event_mask[idxs[regions==loc], w[i], d[i], h_s[i]:h_e[i]] = 1
            event_mask[idxs[regions==loc], w[i], d[i], h_e[i]] = 1
        if np.any(event_mask * anomaly_mask):
#             num_detected_events +=1
            detected_events[i]=1
    return detected_events#num_detected_events
